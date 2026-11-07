#!/usr/bin/python3
# -*- coding:utf-8 -*-
import json
import math
import threading
import rospy
import actionlib
from platform_msgs.msg import UserLogic
from platform_msgs.srv import StopProcessingUserNode, StopProcessingUserNodeResponse
from geometry_msgs.msg import PoseStamped
from krpc_ib2_msgs.srv import MissionStart, MissionFinished, CheckPoints
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal
from ib2_msgs.msg import CtlCommandAction, CtlCommandGoal, CtlCommandResult, CtlStatusType, Navigation


class UserTemplate:

    # พารามิเตอร์ทางวิศวกรรมของภารกิจ 7th Kibo-RPC
    START_DOCK_POS = [0.200, 0.000, 0.000]
    CAMERA_OFFSET_X = 0.14  # กล้องหน้าเยื้องไปทางแกน +X ของตัวหุ่น 0.14 เมตร
    MISSION_TIME_LIMIT_SEC = 300.0
    SAFE_GOAL_RESERVE_SEC = 35.0  # เวลาสำรองขั้นต่ำสำหรับบินเข้า GoalPoint และรายงานผล

    def __init__(self):
        rospy.init_node('krpc_user')

        # ตัวแปรเก็บสถานะตำแหน่งและความเร็วปัจจุบันจากระบบ Navigation
        self.__nav_lock = threading.Lock()
        self.__current_pos = None
        self.__current_quat = None
        self.__current_vel = None
        self.__current_omega = None

        # Action clients
        self.__command = actionlib.SimpleActionClient('/ctl/command', CtlCommandAction)
        self.__checkpoint_client = actionlib.SimpleActionClient('/event/checkpoint_passed', CheckpointPassedAction)

        # Service clients
        self.__mission_start = rospy.ServiceProxy('/event/mission_start', MissionStart)
        self.__mission_finished = rospy.ServiceProxy('/event/mission_finished', MissionFinished)
        self.__checkpoints = rospy.ServiceProxy('/mission/get_checkpoints', CheckPoints)

        # Subscribers
        self.__subscriber_nav = rospy.Subscriber('/sensor_fusion/navigation', Navigation, self.__callback_nav)
        self.__subscriber_start = rospy.Subscriber('/ib2_user/start', UserLogic, self.__callback_start)  # DO NOT REMOVE THIS LINE

        # Service server
        self.__stop_processing_server = rospy.Service('/ib2_user/stop', StopProcessingUserNode, self.__stop_processing)  # DO NOT REMOVE THIS LINE

        self.__started = False
        self.__action_active = False
        self.__checkpoint_action_active = False
        self.__stop_requested = False
        self.__mission_start_time = None

        rospy.set_param('/ib2_user/ready', True)  # DO NOT REMOVE THIS LINE

    # =========================================================================
    # 1. ส่วนจัดการข้อมูลเซ็นเซอร์และคณิตศาสตร์ 3 มิติ (Kinematics & Quaternion Math)
    # =========================================================================

    def __callback_nav(self, data):
        with self.__nav_lock:
            p = data.pose.pose.position
            q = data.pose.pose.orientation
            v = data.twist.linear
            w = data.twist.angular
            self.__current_pos = [float(p.x), float(p.y), float(p.z)]
            self.__current_quat = [float(q.x), float(q.y), float(q.z), float(q.w)]
            self.__current_vel = [float(v.x), float(v.y), float(v.z)]
            self.__current_omega = [float(w.x), float(w.y), float(w.z)]

    def get_current_navigation(self):
        with self.__nav_lock:
            if self.__current_pos is None:
                return None, None, None, None
            return (
                list(self.__current_pos),
                list(self.__current_quat),
                list(self.__current_vel),
                list(self.__current_omega),
            )

    @staticmethod
    def normalize_quaternion(q):
        norm = math.sqrt(sum(c * c for c in q))
        if norm < 1e-9:
            return [0.0, 0.0, 0.0, 1.0]
        return [c / norm for c in q]

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @classmethod
    def quat_to_rot_matrix(cls, q):
        x, y, z, w = cls.normalize_quaternion(q)
        return [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ]

    @classmethod
    def rot_matrix_to_quat(cls, R):
        tr = R[0][0] + R[1][1] + R[2][2]
        if tr > 0.0:
            S = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2][1] - R[1][2]) / S
            y = (R[0][2] - R[2][0]) / S
            z = (R[1][0] - R[0][1]) / S
        elif (R[0][0] > R[1][1]) and (R[0][0] > R[2][2]):
            S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
            w = (R[2][1] - R[1][2]) / S
            x = 0.25 * S
            y = (R[0][1] + R[1][0]) / S
            z = (R[0][2] + R[2][0]) / S
        elif R[1][1] > R[2][2]:
            S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
            w = (R[0][2] - R[2][0]) / S
            x = (R[0][1] + R[1][0]) / S
            y = 0.25 * S
            z = (R[1][2] + R[2][1]) / S
        else:
            S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
            w = (R[1][0] - R[0][1]) / S
            x = (R[0][2] + R[2][0]) / S
            y = (R[1][2] + R[2][1]) / S
            z = 0.25 * S
        return cls.normalize_quaternion([x, y, z, w])

    @classmethod
    def compute_look_at_quaternion(cls, current_pos, marker_pos, ref_quat):
        """
        คำนวณ Quaternion ใหม่เพื่อให้แกน +X ของหุ่นยนต์ (และกล้องหน้าที่เยื้องไป +0.14m บนแกน +X)
        เล็งตรงไปยังพิกัด AR Marker เป๊ะๆ โดยรักษามุม Roll รอบแกนสายตาตาม ref_quat
        """
        if marker_pos is None or current_pos is None:
            return cls.normalize_quaternion(ref_quat)

        dx = marker_pos[0] - current_pos[0]
        dy = marker_pos[1] - current_pos[1]
        dz = marker_pos[2] - current_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-6:
            return cls.normalize_quaternion(ref_quat)

        # แกน +X ใหม่ชี้จากจุดศูนย์กลางหุ่น (และเลนส์กล้อง) พุ่งตรงเข้าหา AR Marker
        x_new = [dx / dist, dy / dist, dz / dist]

        # ดึงแกน +Z อ้างอิงจาก ref_quat เพื่อรักษามุมเอียง (Roll) รอบแกนสายตา
        R_ref = cls.quat_to_rot_matrix(ref_quat)
        z_ref = [R_ref[0][2], R_ref[1][2], R_ref[2][2]]

        # y_new = normalize(z_ref x x_new)
        y_new = [
            z_ref[1] * x_new[2] - z_ref[2] * x_new[1],
            z_ref[2] * x_new[0] - z_ref[0] * x_new[2],
            z_ref[0] * x_new[1] - z_ref[1] * x_new[0],
        ]
        y_norm = math.sqrt(sum(c * c for c in y_new))
        if y_norm < 1e-6:
            return cls.normalize_quaternion(ref_quat)
        y_new = [c / y_norm for c in y_new]

        # z_new = x_new x y_new
        z_new = [
            x_new[1] * y_new[2] - x_new[2] * y_new[1],
            x_new[2] * y_new[0] - x_new[0] * y_new[2],
            x_new[0] * y_new[1] - x_new[1] * y_new[0],
        ]

        R_new = [
            [x_new[0], y_new[0], z_new[0]],
            [x_new[1], y_new[1], z_new[1]],
            [x_new[2], y_new[2], z_new[2]],
        ]
        return cls.rot_matrix_to_quat(R_new)

    # =========================================================================
    # 2. ส่วนควบคุมการเคลื่อนที่แบบระบุพิกัด (Coordinate Movement APIs)
    # =========================================================================

    def __send_ctl_command(self, target_pose, cmd_type_val, timeout_sec=45.0):
        if self.__stop_requested or rospy.is_shutdown():
            return False

        if not self.__command.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr('Action server /ctl/command is unavailable')
            return False

        command_type = CtlStatusType(type=cmd_type_val)
        goal = CtlCommandGoal(target=target_pose, type=command_type)

        self.__action_active = True
        start_wait = rospy.Time.now()
        try:
            self.__command.send_goal(goal)
            while not rospy.is_shutdown() and not self.__stop_requested:
                if self.__command.wait_for_result(rospy.Duration(0.1)):
                    result = self.__command.get_result()
                    if result is not None and result.type == CtlCommandResult.TERMINATE_SUCCESS:
                        return True
                    rospy.logwarn('Movement terminated with status code: %s', getattr(result, 'type', 'None'))
                    return False

                if (rospy.Time.now() - start_wait).to_sec() > timeout_sec:
                    rospy.logwarn('Movement timed out after %.1f sec, cancelling goal', timeout_sec)
                    self.__command.cancel_goal()
                    self.__command.wait_for_result(rospy.Duration(2.0))
                    return False

            if rospy.is_shutdown() or self.__stop_requested:
                self.__command.cancel_goal()
            return False
        finally:
            self.__action_active = False

    def move_to_absolute(self, x, y, z, qx, qy, qz, qw, timeout_sec=45.0, max_retries=2):
        """
        สั่งหุ่นยนต์บินไปยังพิกัดสัมบูรณ์ (x, y, z) และทิศทาง (qx, qy, qz, qw) ในกรอบพิกัดสถานีอวกาศ (iss_body)
        """
        q_norm = self.normalize_quaternion([float(qx), float(qy), float(qz), float(qw)])
        target = PoseStamped()
        target.header.frame_id = 'iss_body'
        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.position.z = float(z)
        target.pose.orientation.x = q_norm[0]
        target.pose.orientation.y = q_norm[1]
        target.pose.orientation.z = q_norm[2]
        target.pose.orientation.w = q_norm[3]

        for attempt in range(1, max_retries + 1):
            if self.__stop_requested or rospy.is_shutdown():
                return False
            target.header.stamp = rospy.Time.now()
            rospy.loginfo(
                '[Absolute Move %d/%d] -> pos=(%.3f, %.3f, %.3f) quat=(%.4f, %.4f, %.4f, %.4f)',
                attempt, max_retries, x, y, z, q_norm[0], q_norm[1], q_norm[2], q_norm[3]
            )
            if self.__send_ctl_command(target, CtlStatusType.MOVE_TO_ABSOLUTE_TARGET, timeout_sec=timeout_sec):
                return True
            rospy.sleep(0.2)
        return False

    def move_to_relative(self, dx, dy, dz, qx=0.0, qy=0.0, qz=0.0, qw=1.0, timeout_sec=20.0):
        """
        สั่งหุ่นยนต์ขยับแบบสัมพัทธ์ (dx, dy, dz) อ้างอิงจากกรอบตัวหุ่นปัจจุบัน (body)
        เช่น dx > 0 คือเดินหน้า, dy > 0 คือขยับซ้าย, dz > 0 คือขยับขึ้นบน
        """
        q_norm = self.normalize_quaternion([float(qx), float(qy), float(qz), float(qw)])
        target = PoseStamped()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = 'body'
        target.pose.position.x = float(dx)
        target.pose.position.y = float(dy)
        target.pose.position.z = float(dz)
        target.pose.orientation.x = q_norm[0]
        target.pose.orientation.y = q_norm[1]
        target.pose.orientation.z = q_norm[2]
        target.pose.orientation.w = q_norm[3]

        rospy.loginfo(
            '[Relative Move] -> d_pos=(%.3f, %.3f, %.3f) d_quat=(%.4f, %.4f, %.4f, %.4f)',
            dx, dy, dz, q_norm[0], q_norm[1], q_norm[2], q_norm[3]
        )
        return self.__send_ctl_command(target, CtlStatusType.MOVE_TO_RELATIVE_TARGET, timeout_sec=timeout_sec)

    def stop_moving(self):
        """
        สั่งเบรกและรักษาระดับท่าทางปัจจุบัน (เทียบเท่า KEEP_POSE)
        """
        target = PoseStamped()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = 'body'
        target.pose.orientation.w = 1.0
        return self.__send_ctl_command(target, CtlStatusType.STOP_MOVING, timeout_sec=10.0)

    def move_and_align_checkpoint(self, checkpoint):
        """
        เคลื่อนที่เข้าสู่จุดตรวจแบบ 2 จังหวะ:
        1) บินเข้าสู่พิกัดจุดตรวจ (Position + Target Attitude)
        2) ตรวจสอบตำแหน่งจริงจาก Navigation หากมีระยะเยื้องศูนย์จะคำนวณปรับมุมกล้องเล็งเข้าหา AR Marker
           และขยับซ้ำเข้าจุดศูนย์กลางเพื่อให้ผ่านเกณฑ์ Lv3 (ระยะ <= 0.1m, มุมกล้อง <= 10 องศา)
        """
        cp_id = checkpoint['checkpoint_id']
        target_pos = checkpoint['position']
        ref_quat = checkpoint['quaternion']
        marker_pos = checkpoint.get('marker_position')
        lv3_radius = checkpoint.get('lv3_radius', 0.10)

        # จังหวะที่ 1: บินเข้าสู่พิกัดเป้าหมายหลัก
        if not self.move_to_absolute(
            target_pos[0], target_pos[1], target_pos[2],
            ref_quat[0], ref_quat[1], ref_quat[2], ref_quat[3],
            timeout_sec=40.0
        ):
            return False

        # จังหวะที่ 2: อ่านค่า Navigation ปัจจุบันเพื่อปรับจูนความแม่นยำระดับ Lv3
        cur_pos, _, _, _ = self.get_current_navigation()
        if cur_pos is not None:
            pos_err = self.euclidean_distance(cur_pos, target_pos)
            # คำนวณ Quaternion ชดเชยให้กล้องหน้า (+0.14m) เล็งตรงเข้าหา AR Marker จากจุดที่ลอยอยู่จริง
            aligned_quat = self.compute_look_at_quaternion(cur_pos, marker_pos, ref_quat)

            # หากตำแหน่งคลาดเคลื่อนเกิน 6 ซม. (ใกล้หลุดกรอบ Lv3 ที่ 10 ซม.) ให้สั่งขยับเข้าจุดศูนย์กลางพร้อมมุมเล็งใหม่
            if pos_err > (lv3_radius * 0.6):
                rospy.loginfo('CP %d position error=%.3fm, refining position & look-at attitude', cp_id, pos_err)
                self.move_to_absolute(
                    target_pos[0], target_pos[1], target_pos[2],
                    aligned_quat[0], aligned_quat[1], aligned_quat[2], aligned_quat[3],
                    timeout_sec=15.0, max_retries=1
                )

        # จังหวะที่ 3: รอและยืนยันว่าหุ่นยนต์ลอยนิ่งอยู่ในโซนครบ 1.0 วินาทีก่อนส่งตรวจคะแนน
        self.wait_for_stability_hold(target_pos, hold_duration_sec=1.1, max_wait_sec=4.0)
        return True

    def wait_for_stability_hold(self, target_pos, hold_duration_sec=1.1, max_wait_sec=4.0):
        """
        หน่วงเวลาและตรวจสอบความเร็วของหุ่นยนต์ให้ลอยนิ่งครบ 1 วินาทีตามกติกาการให้คะแนน
        """
        start_time = rospy.Time.now()
        stable_since = None
        rate = rospy.Rate(20)  # ตรวจสอบที่ความถี่ 20 Hz

        while not rospy.is_shutdown() and not self.__stop_requested:
            now = rospy.Time.now()
            if (now - start_time).to_sec() >= max_wait_sec:
                break

            cur_pos, _, cur_vel, cur_omega = self.get_current_navigation()
            if cur_pos is None or cur_vel is None:
                rospy.sleep(hold_duration_sec)
                break

            speed = math.sqrt(sum(v * v for v in cur_vel))
            ang_speed = math.sqrt(sum(w * w for w in cur_omega))
            dist_err = self.euclidean_distance(cur_pos, target_pos)

            # ถือว่านิ่งเมื่อความเร็วเชิงเส้น < 0.05 m/s, ความเร็วเชิงมุม < 0.08 rad/s และอยู่ในรัศมี 0.30 m (อย่างน้อย Lv1)
            if speed < 0.05 and ang_speed < 0.08 and dist_err <= 0.30:
                if stable_since is None:
                    stable_since = now
                elif (now - stable_since).to_sec() >= hold_duration_sec:
                    return True
            else:
                stable_since = None

            rate.sleep()
        return False

    # =========================================================================
    # 3. ส่วนดึงข้อมูลจุดตรวจและจัดลำดับเส้นทาง (JSON Parser & TSP Route Planner)
    # =========================================================================

    def __get_checkpoint_id(self, checkpoint):
        for key in ('checkpoint_id', 'checkpointid', 'id'):
            if key in checkpoint:
                return int(checkpoint[key])
        return None

    def fetch_and_plan_route(self, max_retries=3):
        """
        ดึงข้อมูล CheckPoints ทั้งหมดจาก /mission/get_checkpoints และจัดเรียงลำดับการบินที่สั้นที่สุด (TSP)
        โดยเริ่มจาก Dock Station (0.2, 0.0, 0.0) ไปครบทุกจุดตรวจ และจบที่ GoalPoint
        """
        rospy.wait_for_service('/mission/get_checkpoints', timeout=10.0)
        payload = None
        for _ in range(max_retries):
            res = self.__checkpoints()
            if res.success:
                try:
                    payload = json.loads(res.message)
                    break
                except Exception as exc:
                    rospy.logwarn('JSON parse error: %s', exc)
            rospy.sleep(0.3)

        if not payload:
            rospy.logerr('Failed to retrieve checkpoints payload')
            return [], None

        raw_list = payload.get('checkpoints', payload.get('Checkpoints', []))
        goal_id = payload.get('goalpoint_id')
        if goal_id is not None:
            goal_id = int(goal_id)

        parsed_cps = []
        for cp in raw_list:
            if not isinstance(cp, dict):
                continue
            cp_id = self.__get_checkpoint_id(cp)
            pos = cp.get('position')
            if cp_id is None or not (isinstance(pos, list) and len(pos) >= 3):
                continue

            quat = [0.0, 0.0, 0.0, 1.0]
            attitude = cp.get('attitude')
            if isinstance(attitude, dict):
                att_q = attitude.get('quaternion')
                if isinstance(att_q, list) and len(att_q) >= 4:
                    quat = [float(att_q[0]), float(att_q[1]), float(att_q[2]), float(att_q[3])]

            marker_pos = None
            marker_pose = cp.get('marker_pose')
            if isinstance(marker_pose, dict):
                m_pos = marker_pose.get('position')
                if isinstance(m_pos, list) and len(m_pos) >= 3:
                    marker_pos = [float(m_pos[0]), float(m_pos[1]), float(m_pos[2])]

            lv3_radius = 0.10
            score_bands = cp.get('score_bands', [])
            for band in score_bands:
                if isinstance(band, dict) and band.get('level') == 3 and 'radius' in band:
                    lv3_radius = float(band['radius'])

            parsed_cps.append({
                'checkpoint_id': cp_id,
                'position': [float(pos[0]), float(pos[1]), float(pos[2])],
                'quaternion': self.normalize_quaternion(quat),
                'marker_position': marker_pos,
                'lv3_radius': lv3_radius,
            })

        if not parsed_cps:
            return [], None

        # แยก GoalPoint ไว้เป็นจุดหมายสุดท้ายเสมอ
        goal_cp = None
        intermediate_cps = []
        for cp in parsed_cps:
            if goal_id is not None and cp['checkpoint_id'] == goal_id:
                goal_cp = cp
            else:
                intermediate_cps.append(cp)

        # หาก goal_id ไม่ได้อยู่ในลิสต์แยก ให้ใช้ตัวสุดท้ายที่มี id ตรงกันหรือสูงสุดเป็น Goal
        if goal_cp is None and intermediate_cps:
            intermediate_cps.sort(key=lambda item: item['checkpoint_id'])
            goal_cp = intermediate_cps.pop(-1)

        # จัดเรียงลำดับด้วย Nearest Neighbor ตามด้วย 2-Opt Optimization
        ordered_route = self.__optimize_tsp_route(self.START_DOCK_POS, intermediate_cps, goal_cp)
        return ordered_route, goal_cp

    def __optimize_tsp_route(self, start_pos, waypoints, goal_cp):
        if not waypoints:
            return []

        # 1. Nearest Neighbor Initialization
        unvisited = list(waypoints)
        route = []
        curr_p = list(start_pos)
        while unvisited:
            nearest_idx = min(
                range(len(unvisited)),
                key=lambda i: self.euclidean_distance(curr_p, unvisited[i]['position'])
            )
            nxt = unvisited.pop(nearest_idx)
            route.append(nxt)
            curr_p = nxt['position']

        # 2. 2-Opt Local Search Refinement (รวมระยะทางเชื่อมต่อไปยัง GoalPoint สุดท้าย)
        def route_cost(seq):
            cost = 0.0
            prev = start_pos
            for item in seq:
                cost += self.euclidean_distance(prev, item['position'])
                prev = item['position']
            if goal_cp is not None:
                cost += self.euclidean_distance(prev, goal_cp['position'])
            return cost

        improved = True
        n = len(route)
        while improved and n >= 3:
            improved = False
            best_cost = route_cost(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                    if route_cost(candidate) + 1e-6 < best_cost:
                        route = candidate
                        best_cost = route_cost(route)
                        improved = True
        return route

    # =========================================================================
    # 4. ส่วนจัดการ Mission Event API และลูปการทำงานหลัก (Mission Flow & Safety)
    # =========================================================================

    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start', timeout=10.0)
        res = self.__mission_start()
        self.__mission_start_time = rospy.Time.now()
        rospy.loginfo('mission_start result: success=%s message=%s', res.success, res.message)
        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished', timeout=10.0)
        res = self.__mission_finished()
        rospy.loginfo('mission_finished result: success=%s message=%s', res.success, res.message)
        return res.success

    def __notify_checkpoint_passed(self, checkpoint_id, max_retries=2):
        if not self.__checkpoint_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr('Action server /event/checkpoint_passed unavailable')
            return False

        for attempt in range(1, max_retries + 1):
            if self.__stop_requested or rospy.is_shutdown():
                return False

            rospy.loginfo('Notifying checkpoint_id=%d passed (attempt %d/%d)', checkpoint_id, attempt, max_retries)
            goal = CheckpointPassedGoal(checkpoint_id=int(checkpoint_id))
            self.__checkpoint_action_active = True
            start_wait = rospy.Time.now()
            try:
                self.__checkpoint_client.send_goal(goal)
                while not rospy.is_shutdown() and not self.__stop_requested:
                    if self.__checkpoint_client.wait_for_result(rospy.Duration(0.1)):
                        result = self.__checkpoint_client.get_result()
                        rospy.loginfo('checkpoint_passed result: %s', result)
                        if result is not None and result.success:
                            return True
                        break
                    if (rospy.Time.now() - start_wait).to_sec() > 10.0:
                        self.__checkpoint_client.cancel_goal()
                        break
            finally:
                self.__checkpoint_action_active = False

            rospy.sleep(0.3)
        return False

    def __remaining_mission_time(self):
        if self.__mission_start_time is None:
            return self.MISSION_TIME_LIMIT_SEC
        elapsed = (rospy.Time.now() - self.__mission_start_time).to_sec()
        return self.MISSION_TIME_LIMIT_SEC - elapsed

    def __run_user_flow(self, msg):
        try:
            if not self.__notify_mission_start():
                return

            # 1. ดึงข้อมูลจุดตรวจและวางแผนลำดับเส้นทางบินทั้งหมด
            route, goal_cp = self.fetch_and_plan_route()
            rospy.loginfo(
                'Planned route: %s | Goal CP: %s',
                [cp['checkpoint_id'] for cp in route],
                goal_cp['checkpoint_id'] if goal_cp else 'None'
            )

            # 2. วนลูปบินไปเก็บแต่ละ CheckPoint ตามลำดับที่สั้นที่สุด
            for cp in route:
                if self.__stop_requested or rospy.is_shutdown():
                    return

                # ตรวจสอบเวลาคงเหลือ หากเหลือน้อยกว่าเวลาสำรอง ให้ข้ามไป GoalPoint ทันที
                if self.__remaining_mission_time() < self.SAFE_GOAL_RESERVE_SEC:
                    rospy.logwarn('Time budget low (%.1fs left), skipping to GoalPoint!', self.__remaining_mission_time())
                    break

                if self.move_and_align_checkpoint(cp):
                    self.__notify_checkpoint_passed(cp['checkpoint_id'])

            # 3. บินไปยัง GoalPoint สุดท้ายและส่งรายงานจบภารกิจ
            if goal_cp is not None and not self.__stop_requested and not rospy.is_shutdown():
                if self.move_and_align_checkpoint(goal_cp):
                    self.__notify_checkpoint_passed(goal_cp['checkpoint_id'])

            if not rospy.is_shutdown() and not self.__stop_requested:
                self.__notify_mission_finished()

        except Exception as exc:
            rospy.logerr('user flow failed: %s', exc)

    def __callback_start(self, msg):
        if self.__started:
            rospy.logwarn('start callback is already processed')
            return
        self.__started = True
        rospy.loginfo('start %s', msg)
        self.__run_user_flow(msg)

    def __stop_processing(self, request):
        try:
            self.__stop_requested = True
            if self.__action_active:
                self.__command.cancel_goal()
            if self.__checkpoint_action_active:
                self.__checkpoint_client.cancel_goal()
            rospy.loginfo('finish processing')
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.SUCCESS)
        except Exception as exc:
            rospy.logerr('stop processing failed: %s', exc)
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.ERROR)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    UserTemplate().run()