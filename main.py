# -*- coding: utf-8 -*-
import pygame
import math
import numpy as np

pygame.init()

# ============================================================
# WINDOW
# ============================================================

WIDTH, HEIGHT = 1400, 850

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Formula Student Vehicle Dynamics Simulator - V1.0  |  AI Race Engineer · Neural Lap Predictor · Aero Sim"
)

clock = pygame.time.Clock()


# ============================================================
# COLORS
# ============================================================

BG = (10, 13, 18)
PANEL = (21, 26, 34)
PANEL2 = (27, 33, 42)

WHITE = (235, 240, 245)
GREY = (145, 155, 165)

GREEN = (70, 220, 130)
CYAN = (65, 210, 230)
BLUE = (80, 150, 255)
YELLOW = (245, 205, 80)
ORANGE = (255, 150, 70)
RED = (240, 80, 80)
PURPLE = (180, 110, 255)


# ============================================================
# FONTS
# ============================================================

font = pygame.font.SysFont("consolas", 17)
small_font = pygame.font.SysFont("consolas", 14)
medium_font = pygame.font.SysFont("consolas", 21)
big_font = pygame.font.SysFont("consolas", 30, bold=True)


# ============================================================
# VEHICLE PARAMETERS
# ============================================================

class VehicleParameters:

    def __init__(self):

        # ----------------------------------------------------
        # VEHICLE GEOMETRY
        # ----------------------------------------------------

        self.wheelbase = 1.55
        self.track = 1.20

        self.mass = 250.0
        self.cg_height = 0.30

        self.front_weight_distribution = 0.50

        # ----------------------------------------------------
        # TYRES
        # ----------------------------------------------------

        self.front_cornering_stiffness = 18000.0
        self.rear_cornering_stiffness = 19000.0

        self.front_mu = 1.55
        self.rear_mu = 1.50

        self.peak_slip_angle = math.radians(6.0)

        # ----------------------------------------------------
        # POWER / BRAKES
        # ----------------------------------------------------

        self.engine_acceleration = 4.0
        self.brake_deceleration = 8.0

        self.drag_coefficient = 0.5
        self.max_speed = 15.0

        # ----------------------------------------------------
        # SUSPENSION
        # ----------------------------------------------------

        self.front_spring_rate = 30000.0
        self.rear_spring_rate = 32000.0

        self.front_damper = 2200.0
        self.rear_damper = 2400.0

        self.front_motion_ratio = 0.90
        self.rear_motion_ratio = 0.90

        self.front_unsprung_mass = 18.0
        self.rear_unsprung_mass = 20.0

        self.front_travel = 0.080
        self.rear_travel = 0.080

        self.road_bump_height = 0.025

        # ----------------------------------------------------
        # BODY ROLL / PITCH
        # ----------------------------------------------------

        self.roll_stiffness_front = 1800.0
        self.roll_stiffness_rear = 1600.0

        # ----------------------------------------------------
        # ROLL CENTRE + ANTI-ROLL BARS
        # ----------------------------------------------------

        self.front_roll_centre = 0.035
        self.rear_roll_centre = 0.045

        self.front_arb_stiffness = 900.0
        self.rear_arb_stiffness = 750.0

        self.pitch_stiffness = 5000.0

    @property
    def b(self):

        return (
            self.front_weight_distribution *
            self.wheelbase
        )

    @property
    def a(self):

        return (
            self.wheelbase -
            self.b
        )


params = VehicleParameters()


# ============================================================
# SUSPENSION AXLE
# ============================================================

class SuspensionAxle:

    def __init__(
        self,
        spring_rate,
        damper,
        motion_ratio,
        travel
    ):

        self.spring_rate = spring_rate
        self.damper = damper
        self.motion_ratio = motion_ratio
        self.travel = travel

        self.displacement = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

        self.force = 0.0

    @property
    def wheel_rate(self):

        return (
            self.spring_rate /
            max(
                0.01,
                self.motion_ratio ** 2
            )
        )

    @property
    def wheel_damper(self):

        return (
            self.damper /
            max(
                0.01,
                self.motion_ratio ** 2
            )
        )

    def ride_frequency(
        self,
        sprung_mass
    ):

        sprung_mass = max(
            1.0,
            sprung_mass
        )

        return (
            math.sqrt(
                self.wheel_rate /
                sprung_mass
            )
            /
            (2.0 * math.pi)
        )

    def damping_ratio(
        self,
        sprung_mass
    ):

        sprung_mass = max(
            1.0,
            sprung_mass
        )

        critical = (
            2.0 *
            math.sqrt(
                self.wheel_rate *
                sprung_mass
            )
        )

        return (
            self.wheel_damper /
            max(
                1.0,
                critical
            )
        )

    def update(
        self,
        road_position,
        dt,
        sprung_mass,
        external_displacement=0.0
    ):

        target = (
            road_position +
            external_displacement
        )

        relative_displacement = (
            self.displacement -
            target
        )

        spring_force = (
            self.wheel_rate *
            relative_displacement
        )

        damper_force = (
            self.wheel_damper *
            self.velocity
        )

        self.force = (
            spring_force +
            damper_force
        )

        self.acceleration = (
            -self.force /
            max(
                1.0,
                sprung_mass
            )
        )

        self.velocity += (
            self.acceleration *
            dt
        )

        self.displacement += (
            self.velocity *
            dt
        )

        self.displacement = max(
            -self.travel,
            min(
                self.travel,
                self.displacement
            )
        )

        self.velocity *= 0.998


# ============================================================
# TYRE
# ============================================================

class Tire:

    def __init__(
        self,
        cornering_stiffness,
        peak_mu,
        peak_slip_angle,
        reference_load
    ):

        self.cornering_stiffness = (
            cornering_stiffness
        )

        self.peak_mu = peak_mu
        self.peak_slip_angle = peak_slip_angle
        self.reference_load = reference_load

    def friction_coefficient(
        self,
        vertical_load
    ):

        vertical_load = max(
            1.0,
            vertical_load
        )

        load_ratio = (
            vertical_load /
            max(
                1.0,
                self.reference_load
            )
        )

        return (
            self.peak_mu *
            load_ratio ** (-0.12)
        )

    def lateral_force(
        self,
        slip_angle,
        vertical_load
    ):

        if vertical_load <= 0:

            return 0.0

        mu = self.friction_coefficient(
            vertical_load
        )

        if (
            abs(slip_angle)
            <=
            self.peak_slip_angle
        ):

            fy = (
                -self.cornering_stiffness *
                slip_angle
            )

        else:

            sign = (
                -1.0
                if slip_angle > 0
                else 1.0
            )

            excess = (
                abs(slip_angle) -
                self.peak_slip_angle
            )

            saturation = math.exp(
                -4.0 * excess
            )

            peak_force = (
                mu *
                vertical_load
            )

            linear_peak = (
                self.cornering_stiffness *
                self.peak_slip_angle
            )

            fy = sign * (
                peak_force -
                (
                    peak_force -
                    linear_peak
                ) *
                saturation
            )

        maximum = (
            mu *
            vertical_load
        )

        return max(
            -maximum,
            min(
                maximum,
                fy
            )
        )


# ============================================================
# VEHICLE
# ============================================================

class Vehicle:

    def __init__(self, p):

        self.p = p

        self.create_tires()
        self.create_suspension()

        self.reset()

    # --------------------------------------------------------
    # TYRES
    # --------------------------------------------------------

    def create_tires(self):

        total_weight = (
            self.p.mass *
            9.81
        )

        front_weight = (
            total_weight *
            self.p.front_weight_distribution
        )

        rear_weight = (
            total_weight -
            front_weight
        )

        self.front_tire = Tire(
            self.p.front_cornering_stiffness,
            self.p.front_mu,
            self.p.peak_slip_angle,
            front_weight / 2
        )

        self.rear_tire = Tire(
            self.p.rear_cornering_stiffness,
            self.p.rear_mu,
            self.p.peak_slip_angle,
            rear_weight / 2
        )

    # --------------------------------------------------------
    # SUSPENSION
    # --------------------------------------------------------

    def create_suspension(self):

        self.front_suspension = SuspensionAxle(
            self.p.front_spring_rate,
            self.p.front_damper,
            self.p.front_motion_ratio,
            self.p.front_travel
        )

        self.rear_suspension = SuspensionAxle(
            self.p.rear_spring_rate,
            self.p.rear_damper,
            self.p.rear_motion_ratio,
            self.p.rear_travel
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.x = WIDTH * 0.35
        self.y = HEIGHT * 0.55

        self.heading = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.ax = 0.0
        self.ay = 0.0

        self.yaw_rate = 0.0
        self.yaw_acceleration = 0.0

        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        self.time = 0.0

        self.pitch_angle = 0.0
        self.roll_angle = 0.0

        self.pitch_rate = 0.0
        self.roll_rate = 0.0

        self.fl_suspension = 0.0
        self.fr_suspension = 0.0
        self.rl_suspension = 0.0
        self.rr_suspension = 0.0

        self.fl_load = 0.0
        self.fr_load = 0.0
        self.rl_load = 0.0
        self.rr_load = 0.0

        self.front_road = 0.0
        self.rear_road = 0.0

        self.roll_moment = 0.0
        self.effective_roll_centre = 0.0
        self.roll_arm = 0.0
        self.total_roll_stiffness = 0.0

        self.front_roll_stiffness_total = 0.0
        self.rear_roll_stiffness_total = 0.0
        self.roll_stiffness_distribution_front = 0.0

        self.lateral_transfer_total = 0.0
        self.geometric_transfer_front = 0.0
        self.geometric_transfer_rear = 0.0
        self.elastic_transfer_total = 0.0
        self.elastic_transfer_front = 0.0
        self.elastic_transfer_rear = 0.0

        self.calculate_wheel_loads()

    # --------------------------------------------------------
    # STATIC LOAD
    # --------------------------------------------------------

    def static_loads(self):

        total_weight = (
            self.p.mass *
            9.81
        )

        front = (
            total_weight *
            self.p.front_weight_distribution
        )

        rear = (
            total_weight -
            front
        )

        return (
            front / 2,
            front / 2,
            rear / 2,
            rear / 2
        )

    # --------------------------------------------------------
    # LOAD TRANSFER
    # --------------------------------------------------------

    def calculate_wheel_loads(self):

        (
            fl_static,
            fr_static,
            rl_static,
            rr_static
        ) = self.static_loads()

        longitudinal_transfer = (
            self.p.mass *
            self.ax *
            self.p.cg_height /
            self.p.wheelbase
        )

        front_longitudinal = -longitudinal_transfer
        rear_longitudinal = longitudinal_transfer

        lateral_transfer_total = (
            self.p.mass *
            abs(self.ay) *
            self.p.cg_height /
            max(0.01, self.p.track)
        )

        geometric_front = (
            self.p.mass *
            abs(self.ay) *
            self.p.front_roll_centre /
            max(0.01, self.p.track)
        )

        geometric_rear = (
            self.p.mass *
            abs(self.ay) *
            self.p.rear_roll_centre /
            max(0.01, self.p.track)
        )

        geometric_total = (
            geometric_front +
            geometric_rear
        )

        elastic_total = max(
            0.0,
            lateral_transfer_total -
            geometric_total
        )

        front_roll_stiffness = (
            self.p.roll_stiffness_front +
            self.p.front_arb_stiffness
        )

        rear_roll_stiffness = (
            self.p.roll_stiffness_rear +
            self.p.rear_arb_stiffness
        )

        total_roll_stiffness = max(
            1.0,
            front_roll_stiffness +
            rear_roll_stiffness
        )

        front_elastic = (
            elastic_total *
            front_roll_stiffness /
            total_roll_stiffness
        )

        rear_elastic = (
            elastic_total -
            front_elastic
        )

        front_lateral = (
            geometric_front +
            front_elastic
        )

        rear_lateral = (
            geometric_rear +
            rear_elastic
        )

        self.lateral_transfer_total = (
            lateral_transfer_total
        )

        self.geometric_transfer_front = (
            geometric_front
        )

        self.geometric_transfer_rear = (
            geometric_rear
        )

        self.elastic_transfer_total = (
            elastic_total
        )

        self.elastic_transfer_front = (
            front_elastic
        )

        self.elastic_transfer_rear = (
            rear_elastic
        )

        self.front_roll_stiffness_total = (
            front_roll_stiffness
        )

        self.rear_roll_stiffness_total = (
            rear_roll_stiffness
        )

        self.roll_stiffness_distribution_front = (
            front_roll_stiffness /
            total_roll_stiffness
        )

        half_front = (
            front_lateral / 2.0
        )

        half_rear = (
            rear_lateral / 2.0
        )

        if self.ay > 0:

            fl = (
                fl_static +
                front_longitudinal -
                half_front
            )

            fr = (
                fr_static +
                front_longitudinal +
                half_front
            )

            rl = (
                rl_static +
                rear_longitudinal -
                half_rear
            )

            rr = (
                rr_static +
                rear_longitudinal +
                half_rear
            )

        else:

            fl = (
                fl_static +
                front_longitudinal +
                half_front
            )

            fr = (
                fr_static +
                front_longitudinal -
                half_front
            )

            rl = (
                rl_static +
                rear_longitudinal +
                half_rear
            )

            rr = (
                rr_static +
                rear_longitudinal -
                half_rear
            )

        minimum_load = 15.0

        self.fl_load = max(
            minimum_load,
            fl
        )

        self.fr_load = max(
            minimum_load,
            fr
        )

        self.rl_load = max(
            minimum_load,
            rl
        )

        self.rr_load = max(
            minimum_load,
            rr
        )

        self.front_axle_load = (
            self.fl_load +
            self.fr_load
        )

        self.rear_axle_load = (
            self.rl_load +
            self.rr_load
        )

        self.total_load = (
            self.front_axle_load +
            self.rear_axle_load
        )

    # --------------------------------------------------------
    # LOAD INTERFACE
    # --------------------------------------------------------

    def calculate_loads(self):

        self.calculate_wheel_loads()

        return (
            self.fl_load,
            self.fr_load,
            self.rl_load,
            self.rr_load
        )

    # --------------------------------------------------------
    # SLIP ANGLES
    # --------------------------------------------------------

    def calculate_slip_angles(self):

        if self.vx <= 0.75:

            return 0.0, 0.0

        front_vy = (
            self.vy +
            self.p.a *
            self.yaw_rate
        )

        rear_vy = (
            self.vy -
            self.p.b *
            self.yaw_rate
        )

        front_alpha = (
            math.atan2(
                front_vy,
                self.vx
            ) -
            self.steering
        )

        rear_alpha = math.atan2(
            rear_vy,
            self.vx
        )

        return (
            front_alpha,
            rear_alpha
        )

    # --------------------------------------------------------
    # FRICTION CIRCLE
    # --------------------------------------------------------

    def friction_circle(
        self,
        fx,
        fy,
        fz,
        tire
    ):

        mu = tire.friction_coefficient(
            fz
        )

        maximum = max(
            1.0,
            mu * fz
        )

        ratio = math.sqrt(
            (fx / maximum) ** 2 +
            (fy / maximum) ** 2
        )

        if ratio > 1.0:

            fx /= ratio
            fy /= ratio

        return (
            fx,
            fy
        )

    # --------------------------------------------------------
    # ROAD PROFILE
    # --------------------------------------------------------

    def road_profile(
        self,
        axle_distance
    ):

        if self.vx < 0.5:

            return 0.0

        phase = (
            self.x /
            180.0
        )

        delay = (
            axle_distance /
            max(
                1.0,
                self.vx * 50
            )
        )

        return (
            self.p.road_bump_height *
            math.sin(
                phase -
                delay * 4.0
            )
        )

    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    def update_pitch(self, dt):

        pitch_target = (
            -self.ax *
            self.p.cg_height /
            max(
                1.0,
                self.p.pitch_stiffness /
                self.p.mass
            )
        )

        pitch_target *= 0.035

        pitch_error = (
            pitch_target -
            self.pitch_angle
        )

        pitch_acceleration = (
            pitch_error * 18.0 -
            self.pitch_rate * 6.0
        )

        self.pitch_rate += (
            pitch_acceleration *
            dt
        )

        self.pitch_angle += (
            self.pitch_rate *
            dt
        )

        self.pitch_angle = max(
            -0.10,
            min(
                0.10,
                self.pitch_angle
            )
        )

    # --------------------------------------------------------
    # ROLL
    # --------------------------------------------------------

    def update_roll(self, dt):

        total_roll_stiffness = (
            self.p.roll_stiffness_front +
            self.p.roll_stiffness_rear +
            self.p.front_arb_stiffness +
            self.p.rear_arb_stiffness
        )

        front_k = (
            self.p.roll_stiffness_front +
            self.p.front_arb_stiffness
        )

        rear_k = (
            self.p.roll_stiffness_rear +
            self.p.rear_arb_stiffness
        )

        ksum = max(
            1.0,
            front_k + rear_k
        )

        effective_rc = (
            self.p.front_roll_centre * front_k +
            self.p.rear_roll_centre * rear_k
        ) / ksum

        roll_arm = (
            self.p.cg_height -
            effective_rc
        )

        roll_moment = (
            self.p.mass *
            self.ay *
            roll_arm
        )

        roll_target = (
            roll_moment /
            max(
                1.0,
                total_roll_stiffness
            )
        )

        roll_target = max(
            -0.15,
            min(
                0.15,
                roll_target
            )
        )

        roll_error = (
            roll_target -
            self.roll_angle
        )

        roll_acceleration = (
            roll_error * 20.0 -
            self.roll_rate * 7.0
        )

        self.roll_rate += (
            roll_acceleration *
            dt
        )

        self.roll_angle += (
            self.roll_rate *
            dt
        )

        self.roll_angle = max(
            -0.15,
            min(
                0.15,
                self.roll_angle
            )
        )

        self.roll_moment = roll_moment
        self.effective_roll_centre = effective_rc
        self.roll_arm = roll_arm
        self.total_roll_stiffness = (
            total_roll_stiffness
        )

    # --------------------------------------------------------
    # FOUR-WHEEL SUSPENSION
    # --------------------------------------------------------

    def update_four_wheel_suspension(
        self,
        dt
    ):

        self.front_road = (
            self.road_profile(0.0)
        )

        self.rear_road = (
            self.road_profile(
                self.p.wheelbase
            )
        )

        total_sprung_mass = (
            self.p.mass -
            self.p.front_unsprung_mass -
            self.p.rear_unsprung_mass
        )

        total_sprung_mass = max(
            20.0,
            total_sprung_mass
        )

        front_sprung_mass = (
            total_sprung_mass *
            self.p.front_weight_distribution
        )

        rear_sprung_mass = (
            total_sprung_mass -
            front_sprung_mass
        )

        # Keep variables explicit for future suspension modelling.
        _ = front_sprung_mass
        _ = rear_sprung_mass

        front_pitch_offset = (
            -self.pitch_angle *
            self.p.a
        )

        rear_pitch_offset = (
            self.pitch_angle *
            self.p.b
        )

        front_roll_offset = (
            self.roll_angle *
            self.p.track /
            2.0
        )

        rear_roll_offset = (
            self.roll_angle *
            self.p.track /
            2.0
        )

        fl_target = (
            self.front_road +
            front_pitch_offset -
            front_roll_offset
        )

        fr_target = (
            self.front_road +
            front_pitch_offset +
            front_roll_offset
        )

        rl_target = (
            self.rear_road +
            rear_pitch_offset -
            rear_roll_offset
        )

        rr_target = (
            self.rear_road +
            rear_pitch_offset +
            rear_roll_offset
        )

        front_rate = max(
            1.0,
            self.front_suspension.wheel_rate
        )

        rear_rate = max(
            1.0,
            self.rear_suspension.wheel_rate
        )

        _ = front_rate
        _ = rear_rate

        front_base = (
            self.front_suspension.displacement
        )

        rear_base = (
            self.rear_suspension.displacement
        )

        front_target = (
            fl_target +
            fr_target
        ) / 2.0

        rear_target = (
            rl_target +
            rr_target
        ) / 2.0

        front_base += (
            front_target -
            front_base
        ) * min(
            1.0,
            dt * 10.0
        )

        rear_base += (
            rear_target -
            rear_base
        ) * min(
            1.0,
            dt * 10.0
        )

        self.front_suspension.displacement = (
            front_base
        )

        self.rear_suspension.displacement = (
            rear_base
        )

        self.fl_suspension = (
            front_base -
            front_roll_offset
        )

        self.fr_suspension = (
            front_base +
            front_roll_offset
        )

        self.rl_suspension = (
            rear_base -
            rear_roll_offset
        )

        self.rr_suspension = (
            rear_base +
            rear_roll_offset
        )

        limit_f = self.p.front_travel
        limit_r = self.p.rear_travel

        self.fl_suspension = max(
            -limit_f,
            min(
                limit_f,
                self.fl_suspension
            )
        )

        self.fr_suspension = max(
            -limit_f,
            min(
                limit_f,
                self.fr_suspension
            )
        )

        self.rl_suspension = max(
            -limit_r,
            min(
                limit_r,
                self.rl_suspension
            )
        )

        self.rr_suspension = max(
            -limit_r,
            min(
                limit_r,
                self.rr_suspension
            )
        )

    # --------------------------------------------------------
    # MAIN PHYSICS UPDATE
    # --------------------------------------------------------

    def update(self, dt):

        self.calculate_wheel_loads()

        (
            fl,
            fr,
            rl,
            rr
        ) = self.calculate_loads()

        front_alpha, rear_alpha = (
            self.calculate_slip_angles()
        )

        # ----------------------------------------------------
        # LATERAL TYRE FORCES
        # ----------------------------------------------------

        fy_fl = self.front_tire.lateral_force(
            front_alpha,
            fl
        )

        fy_fr = self.front_tire.lateral_force(
            front_alpha,
            fr
        )

        fy_rl = self.rear_tire.lateral_force(
            rear_alpha,
            rl
        )

        fy_rr = self.rear_tire.lateral_force(
            rear_alpha,
            rr
        )

        # ----------------------------------------------------
        # LONGITUDINAL FORCES
        # ----------------------------------------------------

        drive_force = (
            self.throttle *
            self.p.mass *
            self.p.engine_acceleration
        )

        brake_force = (
            self.brake *
            self.p.mass *
            self.p.brake_deceleration
        )

        front_brake = (
            brake_force *
            0.55 /
            2.0
        )

        rear_brake = (
            brake_force *
            0.45 /
            2.0
        )

        rear_drive = (
            drive_force /
            2.0
        )

        fx_fl = -front_brake
        fx_fr = -front_brake

        fx_rl = (
            rear_drive -
            rear_brake
        )

        fx_rr = (
            rear_drive -
            rear_brake
        )

        # ----------------------------------------------------
        # FRICTION CIRCLE
        # ----------------------------------------------------

        fx_fl, fy_fl = self.friction_circle(
            fx_fl,
            fy_fl,
            fl,
            self.front_tire
        )

        fx_fr, fy_fr = self.friction_circle(
            fx_fr,
            fy_fr,
            fr,
            self.front_tire
        )

        fx_rl, fy_rl = self.friction_circle(
            fx_rl,
            fy_rl,
            rl,
            self.rear_tire
        )

        fx_rr, fy_rr = self.friction_circle(
            fx_rr,
            fy_rr,
            rr,
            self.rear_tire
        )

        # ----------------------------------------------------
        # TOTAL FORCES
        # ----------------------------------------------------

        total_fx = (
            fx_fl +
            fx_fr +
            fx_rl +
            fx_rr
        )

        total_fy = (
            fy_fl +
            fy_fr +
            fy_rl +
            fy_rr
        )

        # ----------------------------------------------------
        # DRAG
        # ----------------------------------------------------

        drag = (
            self.p.drag_coefficient *
            self.vx *
            abs(self.vx)
        )

        self.ax = (
            total_fx -
            drag
        ) / self.p.mass

        self.vx += (
            self.ax *
            dt
        )

        self.vx = max(
            0.0,
            min(
                self.p.max_speed,
                self.vx
            )
        )

        # ----------------------------------------------------
        # LATERAL ACCELERATION
        # ----------------------------------------------------

        self.ay = (
            total_fy /
            self.p.mass
        )

        self.vy += (
            self.ay *
            dt
        )

        self.vy *= 0.992

        # ----------------------------------------------------
        # YAW DYNAMICS
        # ----------------------------------------------------

        front_force = (
            fy_fl +
            fy_fr
        )

        rear_force = (
            fy_rl +
            fy_rr
        )

        yaw_moment = (
            self.p.a *
            front_force -
            self.p.b *
            rear_force
        )

        yaw_inertia = (
            self.p.mass *
            self.p.wheelbase ** 2 /
            12.0
        )

        self.yaw_acceleration = (
            yaw_moment /
            yaw_inertia
        )

        self.yaw_rate += (
            self.yaw_acceleration *
            dt
        )

        self.yaw_rate *= 0.998

        self.yaw_rate = max(
            -3.0,
            min(
                3.0,
                self.yaw_rate
            )
        )

        self.heading += (
            self.yaw_rate *
            dt
        )

        # ----------------------------------------------------
        # BODY DYNAMICS
        # ----------------------------------------------------

        self.update_pitch(dt)
        self.update_roll(dt)

        # ----------------------------------------------------
        # SUSPENSION
        # ----------------------------------------------------

        self.update_four_wheel_suspension(dt)

        # ----------------------------------------------------
        # WORLD MOVEMENT
        # ----------------------------------------------------

        cos_h = math.cos(
            self.heading
        )

        sin_h = math.sin(
            self.heading
        )

        world_vx = (
            self.vx * cos_h -
            self.vy * sin_h
        )

        world_vy = (
            self.vx * sin_h +
            self.vy * cos_h
        )

        self.x += (
            world_vx *
            50.0 *
            dt
        )

        self.y += (
            world_vy *
            50.0 *
            dt
        )

        # ----------------------------------------------------
        # SCREEN WRAP
        # ----------------------------------------------------

        if self.x < 0:
            self.x = WIDTH

        if self.x > WIDTH:
            self.x = 0

        if self.y < 0:
            self.y = HEIGHT

        if self.y > HEIGHT:
            self.y = 0

        # ----------------------------------------------------
        # LOW SPEED BRAKING
        # ----------------------------------------------------

        if (
            self.vx < 0.08
            and
            self.brake > 0.5
        ):

            self.vx = 0.0
            self.vy *= 0.7
            self.yaw_rate *= 0.7

        self.time += dt

        self.calculate_wheel_loads()

        return {
            "fl": self.fl_load,
            "fr": self.fr_load,
            "rl": self.rl_load,
            "rr": self.rr_load,
            "front_alpha": front_alpha,
            "rear_alpha": rear_alpha
        }


vehicle = Vehicle(params)


# ============================================================
# TELEMETRY
# ============================================================

telemetry_time = []
telemetry_speed = []
telemetry_ax = []
telemetry_ay = []
telemetry_yaw = []
telemetry_steering = []

telemetry_front_suspension = []
telemetry_rear_suspension = []

telemetry_fl_load = []
telemetry_fr_load = []
telemetry_rl_load = []
telemetry_rr_load = []

telemetry_pitch = []
telemetry_roll = []

MAX_POINTS = 720


def record_telemetry():

    telemetry_time.append(
        vehicle.time
    )

    telemetry_speed.append(
        vehicle.vx * 3.6
    )

    telemetry_ax.append(
        vehicle.ax / 9.81
    )

    telemetry_ay.append(
        vehicle.ay / 9.81
    )

    telemetry_yaw.append(
        math.degrees(
            vehicle.yaw_rate
        )
    )

    telemetry_steering.append(
        math.degrees(
            vehicle.steering
        )
    )

    telemetry_front_suspension.append(
        (
            vehicle.fl_suspension +
            vehicle.fr_suspension
        ) / 2.0 * 1000
    )

    telemetry_rear_suspension.append(
        (
            vehicle.rl_suspension +
            vehicle.rr_suspension
        ) / 2.0 * 1000
    )

    telemetry_fl_load.append(
        vehicle.fl_load
    )

    telemetry_fr_load.append(
        vehicle.fr_load
    )

    telemetry_rl_load.append(
        vehicle.rl_load
    )

    telemetry_rr_load.append(
        vehicle.rr_load
    )

    telemetry_pitch.append(
        math.degrees(
            vehicle.pitch_angle
        )
    )

    telemetry_roll.append(
        math.degrees(
            vehicle.roll_angle
        )
    )

    if len(telemetry_time) > MAX_POINTS:

        arrays = [
            telemetry_time,
            telemetry_speed,
            telemetry_ax,
            telemetry_ay,
            telemetry_yaw,
            telemetry_steering,
            telemetry_front_suspension,
            telemetry_rear_suspension,
            telemetry_fl_load,
            telemetry_fr_load,
            telemetry_rl_load,
            telemetry_rr_load,
            telemetry_pitch,
            telemetry_roll
        ]

        for data in arrays:
            data.pop(0)


def reset_all():

    vehicle.reset()

    arrays = [
        telemetry_time,
        telemetry_speed,
        telemetry_ax,
        telemetry_ay,
        telemetry_yaw,
        telemetry_steering,
        telemetry_front_suspension,
        telemetry_rear_suspension,
        telemetry_fl_load,
        telemetry_fr_load,
        telemetry_rl_load,
        telemetry_rr_load,
        telemetry_pitch,
        telemetry_roll
    ]

    for data in arrays:
        data.clear()


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_text(
    text,
    x,
    y,
    color=WHITE,
    f=font
):

    surface = f.render(
        str(text),
        True,
        color
    )

    screen.blit(
        surface,
        (x, y)
    )


def panel(
    rect,
    color=PANEL
):

    rect = pygame.Rect(rect)

    pygame.draw.rect(
        screen,
        color,
        rect,
        border_radius=10
    )

    pygame.draw.rect(
        screen,
        (48, 56, 68),
        rect,
        1,
        border_radius=10
    )


def draw_graph(
    rect,
    data,
    min_value,
    max_value,
    title,
    line_color=CYAN
):

    rect = pygame.Rect(rect)

    pygame.draw.rect(
        screen,
        (14, 18, 24),
        rect,
        border_radius=7
    )

    pygame.draw.rect(
        screen,
        (48, 55, 65),
        rect,
        1,
        border_radius=7
    )

    draw_text(
        title,
        rect.x + 10,
        rect.y + 7,
        WHITE,
        small_font
    )

    if len(data) < 2:
        return

    left = rect.x + 35
    right = rect.right - 10

    top = rect.y + 28
    bottom = rect.bottom - 12

    if max_value == min_value:

        max_value += 1
        min_value -= 1

    points = []

    for i, value in enumerate(data):

        x = (
            left +
            i /
            (len(data) - 1) *
            (right - left)
        )

        normalized = (
            value -
            min_value
        ) / (
            max_value -
            min_value
        )

        normalized = max(
            0,
            min(
                1,
                normalized
            )
        )

        y = (
            bottom -
            normalized *
            (bottom - top)
        )

        points.append(
            (
                int(x),
                int(y)
            )
        )

    pygame.draw.lines(
        screen,
        line_color,
        False,
        points,
        2
    )


# ============================================================
# CAR DRAWING
# ============================================================

def draw_car():

    length = 105
    width = 45

    cx = vehicle.x
    cy = vehicle.y

    c = math.cos(
        vehicle.heading
    )

    s = math.sin(
        vehicle.heading
    )

    def transform(px, py):

        return (
            cx + px * c - py * s,
            cy + px * s + py * c
        )

    corners = [
        transform(
            length / 2,
            width / 2
        ),
        transform(
            length / 2,
            -width / 2
        ),
        transform(
            -length / 2,
            -width / 2
        ),
        transform(
            -length / 2,
            width / 2
        )
    ]

    pygame.draw.polygon(
        screen,
        (55, 65, 80),
        corners
    )

    pygame.draw.polygon(
        screen,
        WHITE,
        corners,
        2
    )

    nose = [
        transform(
            length / 2,
            0
        ),
        transform(
            length / 2 - 30,
            -width / 2
        ),
        transform(
            length / 2 - 30,
            width / 2
        )
    ]

    pygame.draw.polygon(
        screen,
        CYAN,
        nose
    )

    cockpit = [
        transform(
            12,
            -15
        ),
        transform(
            12,
            15
        ),
        transform(
            -28,
            12
        ),
        transform(
            -28,
            -12
        )
    ]

    pygame.draw.polygon(
        screen,
        (7, 9, 13),
        cockpit
    )

    hx, hy = transform(
        -3,
        0
    )

    pygame.draw.circle(
        screen,
        YELLOW,
        (
            int(hx),
            int(hy)
        ),
        7
    )

    for px, py in [
        (32, -28),
        (32, 28),
        (-32, -28),
        (-32, 28)
    ]:

        wx, wy = transform(
            px,
            py
        )

        pygame.draw.rect(
            screen,
            (8, 8, 10),
            (
                int(wx - 8),
                int(wy - 13),
                16,
                26
            ),
            border_radius=3
        )


# ============================================================
# DRIVE SCREEN
# ============================================================

def draw_drive():

    screen.fill(BG)

    draw_text(
        "FORMULA STUDENT VEHICLE DYNAMICS SIMULATOR",
        25,
        18,
        WHITE,
        big_font
    )

    draw_text(
        "V0.5  |  T=Telem  H=Handling  L=Design  J=Susp  K=TrackSim",
        28,
        55,
        CYAN,
        small_font
    )

    pygame.draw.rect(
        screen,
        (16, 20, 26),
        (
            20,
            90,
            750,
            740
        ),
        border_radius=12
    )

    for x in range(
        30,
        760,
        80
    ):

        pygame.draw.line(
            screen,
            (28, 34, 42),
            (
                x,
                100
            ),
            (
                x,
                820
            ),
            1
        )

    for y in range(
        100,
        820,
        70
    ):

        pygame.draw.line(
            screen,
            (28, 34, 42),
            (
                30,
                y
            ),
            (
                760,
                y
            ),
            1
        )

    draw_car()

    panel(
        (
            790,
            20,
            590,
            285
        )
    )

    draw_text(
        "LIVE VEHICLE STATE",
        815,
        35,
        CYAN,
        medium_font
    )

    rows = [

        (
            "Speed",
            f"{vehicle.vx * 3.6:6.1f} km/h"
        ),

        (
            "Longitudinal G",
            f"{vehicle.ax / 9.81:+6.2f} G"
        ),

        (
            "Lateral G",
            f"{vehicle.ay / 9.81:+6.2f} G"
        ),

        (
            "Yaw Rate",
            f"{math.degrees(vehicle.yaw_rate):+6.1f} deg/s"
        ),

        (
            "Pitch",
            f"{math.degrees(vehicle.pitch_angle):+6.2f} deg"
        ),

        (
            "Roll",
            f"{math.degrees(vehicle.roll_angle):+6.2f} deg"
        ),

        (
            "Steering",
            f"{math.degrees(vehicle.steering):+6.1f} deg"
        ),

        (
            "CG Height",
            f"{params.cg_height:.2f} m"
        )
    ]

    y = 72

    for label, value in rows:

        draw_text(
            label,
            815,
            y,
            GREY,
            small_font
        )

        draw_text(
            value,
            1040,
            y,
            WHITE,
            small_font
        )

        y += 27

    panel(
        (
            790,
            325,
            590,
            280
        )
    )

    draw_text(
        "4-WHEEL LOAD TRANSFER",
        815,
        340,
        YELLOW,
        medium_font
    )

    car_x = 1090
    car_y = 475

    pygame.draw.rect(
        screen,
        (40, 48, 58),
        (
            car_x - 70,
            car_y - 100,
            140,
            200
        ),
        border_radius=18
    )

    pygame.draw.rect(
        screen,
        (8, 10, 14),
        (
            car_x - 52,
            car_y - 65,
            104,
            80
        ),
        border_radius=15
    )

    loads = [
        vehicle.fl_load,
        vehicle.fr_load,
        vehicle.rl_load,
        vehicle.rr_load
    ]

    max_load = max(
        loads + [1]
    )

    def load_color(load):

        ratio = (
            load /
            max_load
        )

        if ratio > 0.85:
            return RED

        if ratio > 0.65:
            return ORANGE

        return GREEN

    wheel_positions = {

        "FL": (
            car_x - 75,
            car_y - 65
        ),

        "FR": (
            car_x + 75,
            car_y - 65
        ),

        "RL": (
            car_x - 75,
            car_y + 65
        ),

        "RR": (
            car_x + 75,
            car_y + 65
        )
    }

    wheel_data = [

        (
            "FL",
            vehicle.fl_load
        ),

        (
            "FR",
            vehicle.fr_load
        ),

        (
            "RL",
            vehicle.rl_load
        ),

        (
            "RR",
            vehicle.rr_load
        )
    ]

    for name, load in wheel_data:

        wx, wy = wheel_positions[
            name
        ]

        pygame.draw.rect(
            screen,
            load_color(load),
            (
                wx - 18,
                wy - 28,
                36,
                56
            ),
            border_radius=6
        )

        draw_text(
            name,
            wx - 13,
            wy - 20,
            BG,
            small_font
        )

        draw_text(
            f"{load:.0f}",
            wx - 15,
            wy + 2,
            BG,
            small_font
        )

    draw_text(
        "FL",
        820,
        400,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.fl_load:.0f} N",
        860,
        400,
        WHITE,
        small_font
    )

    draw_text(
        "FR",
        820,
        430,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.fr_load:.0f} N",
        860,
        430,
        WHITE,
        small_font
    )

    draw_text(
        "RL",
        820,
        460,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.rl_load:.0f} N",
        860,
        460,
        WHITE,
        small_font
    )

    draw_text(
        "RR",
        820,
        490,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.rr_load:.0f} N",
        860,
        490,
        WHITE,
        small_font
    )

    draw_text(
        "Front Axle",
        820,
        535,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.front_axle_load:.0f} N",
        920,
        535,
        GREEN,
        small_font
    )

    draw_text(
        "Rear Axle",
        820,
        565,
        GREY,
        small_font
    )

    draw_text(
        f"{vehicle.rear_axle_load:.0f} N",
        920,
        565,
        GREEN,
        small_font
    )

    panel(
        (
            790,
            625,
            590,
            205
        )
    )

    draw_text(
        "SUSPENSION RESPONSE",
        815,
        640,
        GREEN,
        medium_font
    )

    suspension_rows = [

        (
            "FL Travel",
            f"{vehicle.fl_suspension * 1000:+.1f} mm"
        ),

        (
            "FR Travel",
            f"{vehicle.fr_suspension * 1000:+.1f} mm"
        ),

        (
            "RL Travel",
            f"{vehicle.rl_suspension * 1000:+.1f} mm"
        ),

        (
            "RR Travel",
            f"{vehicle.rr_suspension * 1000:+.1f} mm"
        ),

        (
            "Pitch / Roll",
            f"{math.degrees(vehicle.pitch_angle):+.2f} / "
            f"{math.degrees(vehicle.roll_angle):+.2f} deg"
        )
    ]

    y = 675

    for label, value in suspension_rows:

        draw_text(
            label,
            815,
            y,
            GREY,
            small_font
        )

        draw_text(
            value,
            1010,
            y,
            WHITE,
            small_font
        )

        y += 27


# ============================================================
# TELEMETRY SCREEN
# ============================================================

def draw_telemetry():

    screen.fill(BG)

    draw_text(
        "LIVE ENGINEERING TELEMETRY",
        30,
        25,
        WHITE,
        big_font
    )

    draw_text(
        "T / ESC = return to DRIVE",
        33,
        65,
        CYAN,
        small_font
    )

    draw_graph(
        (
            30,
            105,
            650,
            180
        ),
        telemetry_speed,
        0,
        max(
            20,
            max(
                telemetry_speed
            ) + 2
            if telemetry_speed
            else 20
        ),
        "VEHICLE SPEED [km/h]",
        BLUE
    )

    draw_graph(
        (
            700,
            105,
            650,
            180
        ),
        telemetry_ax,
        -1.5,
        1.5,
        "LONGITUDINAL ACCELERATION [G]",
        ORANGE
    )

    draw_graph(
        (
            30,
            305,
            650,
            180
        ),
        telemetry_ay,
        -1.5,
        1.5,
        "LATERAL ACCELERATION [G]",
        GREEN
    )

    draw_graph(
        (
            700,
            305,
            650,
            180
        ),
        telemetry_yaw,
        -180,
        180,
        "YAW RATE [deg/s]",
        PURPLE
    )

    draw_graph(
        (
            30,
            505,
            650,
            180
        ),
        telemetry_fl_load,
        0,
        max(
            1000,
            max(
                telemetry_fl_load
            ) + 100
            if telemetry_fl_load
            else 1000
        ),
        "FRONT LEFT LOAD [N]",
        CYAN
    )

    draw_graph(
        (
            700,
            505,
            650,
            180
        ),
        telemetry_fr_load,
        0,
        max(
            1000,
            max(
                telemetry_fr_load
            ) + 100
            if telemetry_fr_load
            else 1000
        ),
        "FRONT RIGHT LOAD [N]",
        RED
    )

    draw_graph(
        (
            30,
            705,
            650,
            120
        ),
        telemetry_pitch,
        -8,
        8,
        "PITCH [deg]",
        YELLOW
    )

    draw_graph(
        (
            700,
            705,
            650,
            120
        ),
        telemetry_roll,
        -8,
        8,
        "ROLL [deg]",
        ORANGE
    )


# ============================================================
# HANDLING TEST
# ============================================================

class HandlingTest:

    def __init__(self):

        self.reset()

    def reset(self):

        self.speed_target = 8.0
        self.duration = 12.0

        self.times = []
        self.ay = []
        self.steering = []

        self.front_slip = []
        self.rear_slip = []

        self.fl_load = []
        self.fr_load = []
        self.rl_load = []
        self.rr_load = []

        self.result = None

    def run(self):

        self.reset()

        test_vehicle = Vehicle(
            params
        )

        dt = 1 / 120

        steps = int(
            self.duration /
            dt
        )

        for i in range(steps):

            t = i * dt

            # ------------------------------------------------
            # STEERING INPUT
            # ------------------------------------------------

            if t < 2.0:

                steering = 0.0

            else:

                phase = (
                    t - 2.0
                ) / 10.0

                steering = (
                    math.radians(18.0) *
                    min(
                        1.0,
                        phase
                    )
                )

            test_vehicle.throttle = 0.4
            test_vehicle.brake = 0.0
            test_vehicle.steering = steering

            if (
                test_vehicle.vx <
                self.speed_target
            ):

                test_vehicle.throttle = 0.5

            test_vehicle.update(
                dt
            )

            fa, ra = (
                test_vehicle.calculate_slip_angles()
            )

            self.times.append(
                t
            )

            self.ay.append(
                test_vehicle.ay /
                9.81
            )

            self.steering.append(
                math.degrees(
                    steering
                )
            )

            self.front_slip.append(
                abs(
                    math.degrees(
                        fa
                    )
                )
            )

            self.rear_slip.append(
                abs(
                    math.degrees(
                        ra
                    )
                )
            )

            self.fl_load.append(
                test_vehicle.fl_load
            )

            self.fr_load.append(
                test_vehicle.fr_load
            )

            self.rl_load.append(
                test_vehicle.rl_load
            )

            self.rr_load.append(
                test_vehicle.rr_load
            )

        # ====================================================
        # RESULTS
        # ====================================================

        peak_g = max(
            (
                abs(x)
                for x in self.ay
            ),
            default=0.0
        )

        max_front = max(
            self.front_slip,
            default=0.0
        )

        max_rear = max(
            self.rear_slip,
            default=0.0
        )

        # ----------------------------------------------------
        # UNDERSTEER GRADIENT
        # ----------------------------------------------------

        valid_x = []
        valid_y = []

        for steer, ay in zip(
            self.steering,
            self.ay
        ):

            if (
                abs(ay) > 0.15
                and
                abs(ay) <
                max(
                    0.2,
                    peak_g * 0.75
                )
            ):

                valid_x.append(
                    abs(ay)
                )

                valid_y.append(
                    steer
                )

        gradient = 0.0
        intercept = 0.0

        if len(valid_x) >= 2:

            xm = (
                sum(valid_x) /
                len(valid_x)
            )

            ym = (
                sum(valid_y) /
                len(valid_y)
            )

            den = sum(
                (
                    v - xm
                ) ** 2
                for v in valid_x
            )

            if den > 1e-9:

                gradient = (
                    sum(
                        (
                            valid_x[i] - xm
                        ) *
                        (
                            valid_y[i] - ym
                        )
                        for i in range(
                            len(valid_x)
                        )
                    )
                    /
                    den
                )

                intercept = (
                    ym -
                    gradient * xm
                )

        # ----------------------------------------------------
        # HANDLING BALANCE
        # ----------------------------------------------------

        slip_difference = (
            max_front -
            max_rear
        )

        if slip_difference > 0.5:

            balance = "FRONT LIMITED"

        elif slip_difference < -0.5:

            balance = "REAR LIMITED"

        else:

            balance = "BALANCED"

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        self.result = {

            "peak_g": peak_g,

            "front_slip": max_front,

            "rear_slip": max_rear,

            "gradient": gradient,

            "intercept": intercept,

            "balance": balance,

            "max_fl": max(
                self.fl_load,
                default=0
            ),

            "max_fr": max(
                self.fr_load,
                default=0
            ),

            "max_rl": max(
                self.rl_load,
                default=0
            ),

            "max_rr": max(
                self.rr_load,
                default=0
            )
        }


handling_test = HandlingTest()


# ============================================================
# HANDLING RESULT
# ============================================================

def draw_handling_result():

    screen.fill(BG)

    draw_text(
        "HANDLING TEST",
        35,
        25,
        WHITE,
        big_font
    )

    draw_text(
        "ENTER / ESC = Back to DRIVE",
        38,
        65,
        CYAN,
        small_font
    )

    result = handling_test.result

    if result is None:

        draw_text(
            "No handling result available.",
            40,
            120,
            RED,
            font
        )

        return

    panel(
        (
            35,
            105,
            410,
            330
        )
    )

    draw_text(
        "RESULTS",
        60,
        130,
        GREEN,
        medium_font
    )

    values = [

        (
            "Peak Lateral G",
            f"{result['peak_g']:.2f} G"
        ),

        (
            "Max Front Slip",
            f"{result['front_slip']:.2f} deg"
        ),

        (
            "Max Rear Slip",
            f"{result['rear_slip']:.2f} deg"
        ),

        (
            "Gradient",
            f"{result['gradient']:.2f} deg/G"
        ),

        (
            "Balance",
            result["balance"]
        )
    ]

    y = 180

    for label, value in values:

        draw_text(
            label,
            60,
            y,
            GREY,
            font
        )

        draw_text(
            value,
            260,
            y,
            YELLOW,
            font
        )

        y += 38

    draw_text(
        "Peak Wheel Loads",
        60,
        385,
        CYAN,
        small_font
    )

    draw_text(
        f"FL {result['max_fl']:.0f} N",
        60,
        408,
        WHITE,
        small_font
    )

    draw_text(
        f"FR {result['max_fr']:.0f} N",
        160,
        408,
        WHITE,
        small_font
    )

    draw_text(
        f"RL {result['max_rl']:.0f} N",
        260,
        408,
        WHITE,
        small_font
    )

    draw_text(
        f"RR {result['max_rr']:.0f} N",
        360,
        408,
        WHITE,
        small_font
    )

    draw_graph(
        (
            480,
            105,
            420,
            280
        ),
        handling_test.ay,
        -1.5,
        1.5,
        "LATERAL G",
        GREEN
    )

    draw_graph(
        (
            935,
            105,
            420,
            280
        ),
        handling_test.steering,
        0,
        20,
        "STEERING ANGLE",
        CYAN
    )

    draw_graph(
        (
            35,
            470,
            640,
            280
        ),
        handling_test.fl_load,
        0,
        max(
            1000,
            max(
                handling_test.fl_load
            ) + 100
            if handling_test.fl_load
            else 1000
        ),
        "FRONT LEFT LOAD",
        BLUE
    )

    draw_graph(
        (
            700,
            470,
            655,
            280
        ),
        handling_test.fr_load,
        0,
        max(
            1000,
            max(
                handling_test.fr_load
            ) + 100
            if handling_test.fr_load
            else 1000
        ),
        "FRONT RIGHT LOAD",
        RED
    )


# ============================================================
# DESIGN LAB
# ============================================================

design_parameters = [

    "CG Height",
    "Mass",
    "Front Weight Distribution",
    "Front C-alpha",
    "Rear C-alpha",
    "Front Mu",
    "Rear Mu"
]

design_selected = 0


def design_value(name):

    if name == "CG Height":

        return (
            f"{params.cg_height:.2f} m"
        )

    if name == "Mass":

        return (
            f"{params.mass:.0f} kg"
        )

    if name == "Front Weight Distribution":

        return (
            f"{params.front_weight_distribution * 100:.0f}%"
        )

    if name == "Front C-alpha":

        return (
            f"{params.front_cornering_stiffness:.0f} N/rad"
        )

    if name == "Rear C-alpha":

        return (
            f"{params.rear_cornering_stiffness:.0f} N/rad"
        )

    if name == "Front Mu":

        return (
            f"{params.front_mu:.2f}"
        )

    if name == "Rear Mu":

        return (
            f"{params.rear_mu:.2f}"
        )

    return ""


def modify_design(
    name,
    direction
):

    if name == "CG Height":

        params.cg_height += (
            0.02 *
            direction
        )

        params.cg_height = max(
            0.15,
            min(
                0.50,
                params.cg_height
            )
        )

    elif name == "Mass":

        params.mass += (
            10 *
            direction
        )

        params.mass = max(
            150,
            min(
                400,
                params.mass
            )
        )

    elif name == "Front Weight Distribution":

        params.front_weight_distribution += (
            0.05 *
            direction
        )

        params.front_weight_distribution = max(
            0.35,
            min(
                0.65,
                params.front_weight_distribution
            )
        )

    elif name == "Front C-alpha":

        params.front_cornering_stiffness += (
            2000 *
            direction
        )

        params.front_cornering_stiffness = max(
            10000,
            min(
                30000,
                params.front_cornering_stiffness
            )
        )

    elif name == "Rear C-alpha":

        params.rear_cornering_stiffness += (
            2000 *
            direction
        )

        params.rear_cornering_stiffness = max(
            10000,
            min(
                30000,
                params.rear_cornering_stiffness
            )
        )

    elif name == "Front Mu":

        params.front_mu += (
            0.1 *
            direction
        )

        params.front_mu = max(
            1.0,
            min(
                2.0,
                params.front_mu
            )
        )

    elif name == "Rear Mu":

        params.rear_mu += (
            0.1 *
            direction
        )

        params.rear_mu = max(
            1.0,
            min(
                2.0,
                params.rear_mu
            )
        )

    vehicle.create_tires()


def draw_design_lab():

    screen.fill(BG)

    draw_text(
        "VEHICLE DESIGN LAB",
        35,
        25,
        WHITE,
        big_font
    )

    draw_text(
        "V0.4.3  |  VEHICLE PARAMETER SENSITIVITY",
        38,
        65,
        CYAN,
        small_font
    )

    panel(
        (
            35,
            110,
            720,
            620
        )
    )

    draw_text(
        "VEHICLE PARAMETERS",
        60,
        135,
        GREEN,
        medium_font
    )

    for i, name in enumerate(
        design_parameters
    ):

        selected = (
            i ==
            design_selected
        )

        y = (
            190 +
            i * 65
        )

        draw_text(
            ">"
            if selected
            else " ",
            55,
            y,
            GREEN,
            medium_font
        )

        draw_text(
            name,
            90,
            y,
            WHITE if selected else GREY,
            medium_font
        )

        draw_text(
            design_value(name),
            470,
            y,
            YELLOW if selected else WHITE,
            medium_font
        )

    draw_text(
        "↑ / ↓ Select",
        60,
        610,
        CYAN,
        small_font
    )

    draw_text(
        "← / → Adjust",
        60,
        635,
        CYAN,
        small_font
    )

    draw_text(
        "SPACE = Sensitivity Sweep",
        60,
        660,
        CYAN,
        small_font
    )

    draw_text(
        "R = Reset Parameters",
        60,
        685,
        CYAN,
        small_font
    )

    panel(
        (
            790,
            110,
            575,
            620
        )
    )

    draw_text(
        "CURRENT VEHICLE",
        820,
        135,
        YELLOW,
        medium_font
    )

    current = [

        (
            "Wheelbase",
            f"{params.wheelbase:.2f} m"
        ),

        (
            "Track",
            f"{params.track:.2f} m"
        ),

        (
            "Mass",
            f"{params.mass:.0f} kg"
        ),

        (
            "CG Height",
            f"{params.cg_height:.2f} m"
        ),

        (
            "Front Distribution",
            f"{params.front_weight_distribution * 100:.0f}%"
        ),

        (
            "Front C-alpha",
            f"{params.front_cornering_stiffness:.0f}"
        ),

        (
            "Rear C-alpha",
            f"{params.rear_cornering_stiffness:.0f}"
        ),

        (
            "Front Mu",
            f"{params.front_mu:.2f}"
        ),

        (
            "Rear Mu",
            f"{params.rear_mu:.2f}"
        )
    ]

    y = 185

    for label, value in current:

        draw_text(
            label,
            820,
            y,
            GREY,
            small_font
        )

        draw_text(
            value,
            1080,
            y,
            WHITE,
            small_font
        )

        y += 34

    draw_text(
        "ENTER = Drive",
        820,
        625,
        CYAN,
        small_font
    )

    draw_text(
        "ESC = Back",
        820,
        650,
        CYAN,
        small_font
    )


# ============================================================
# SENSITIVITY STUDY
# ============================================================

sweep_results = []

sweep_parameter = 0

sweep_parameters = [

    "CG Height",
    "Mass",
    "Front Weight Distribution",
    "Front C-alpha",
    "Rear C-alpha",
    "Front Mu",
    "Rear Mu"
]


def get_param_value(name):

    if name == "CG Height":
        return params.cg_height

    if name == "Mass":
        return params.mass

    if name == "Front Weight Distribution":
        return params.front_weight_distribution

    if name == "Front C-alpha":
        return params.front_cornering_stiffness

    if name == "Rear C-alpha":
        return params.rear_cornering_stiffness

    if name == "Front Mu":
        return params.front_mu

    if name == "Rear Mu":
        return params.rear_mu

    return 0.0


def set_param_value(
    name,
    value
):

    if name == "CG Height":
        params.cg_height = value

    elif name == "Mass":
        params.mass = value

    elif name == "Front Weight Distribution":
        params.front_weight_distribution = value

    elif name == "Front C-alpha":
        params.front_cornering_stiffness = value

    elif name == "Rear C-alpha":
        params.rear_cornering_stiffness = value

    elif name == "Front Mu":
        params.front_mu = value

    elif name == "Rear Mu":
        params.rear_mu = value

    vehicle.create_tires()


def run_handling_for_current():

    test = HandlingTest()

    test.run()

    return test.result


def run_sensitivity():

    global sweep_results

    sweep_results = []

    name = sweep_parameters[
        sweep_parameter
    ]

    original = get_param_value(
        name
    )

    if name == "CG Height":

        values = np.arange(
            0.20,
            0.401,
            0.05
        )

    elif name == "Mass":

        values = np.arange(
            200,
            301,
            25
        )

    elif name == "Front Weight Distribution":

        values = np.arange(
            0.40,
            0.601,
            0.05
        )

    elif name == "Front C-alpha":

        values = np.arange(
            14000,
            22001,
            2000
        )

    elif name == "Rear C-alpha":

        values = np.arange(
            14000,
            22001,
            2000
        )

    elif name == "Front Mu":

        values = np.arange(
            1.3,
            1.701,
            0.1
        )

    else:

        values = np.arange(
            1.3,
            1.701,
            0.1
        )

    for value in values:

        set_param_value(
            name,
            float(value)
        )

        result = (
            run_handling_for_current()
        )

        sweep_results.append(
            {
                "value": float(value),
                "peak_g": result["peak_g"],
                "gradient": result["gradient"]
            }
        )

    set_param_value(
        name,
        original
    )

    vehicle.create_tires()


def draw_sweep_result():

    screen.fill(BG)

    draw_text(
        "PARAMETER SENSITIVITY STUDY",
        30,
        25,
        WHITE,
        big_font
    )

    draw_text(
        "ENTER / ESC = Back to Design Lab",
        33,
        65,
        CYAN,
        small_font
    )

    if not sweep_results:

        draw_text(
            "No sweep results.",
            40,
            120,
            RED,
            font
        )

        return

    name = sweep_parameters[
        sweep_parameter
    ]

    peak_g = [
        r["peak_g"]
        for r in sweep_results
    ]

    gradients = [
        r["gradient"]
        for r in sweep_results
    ]

    peak_min = min(peak_g)
    peak_max = max(peak_g)

    if peak_min == peak_max:

        peak_min -= 0.1
        peak_max += 0.1

    else:

        peak_min -= 0.1
        peak_max += 0.1

    gradient_min = min(gradients)
    gradient_max = max(gradients)

    if gradient_min == gradient_max:

        gradient_min -= 1.0
        gradient_max += 1.0

    else:

        gradient_min -= 10
        gradient_max += 10

    draw_graph(
        (
            35,
            110,
            640,
            300
        ),
        peak_g,
        peak_min,
        peak_max,
        f"PEAK LATERAL G vs {name}",
        GREEN
    )

    draw_graph(
        (
            720,
            110,
            640,
            300
        ),
        gradients,
        gradient_min,
        gradient_max,
        f"UNDERSTEER GRADIENT vs {name}",
        ORANGE
    )

    panel(
        (
            35,
            450,
            1325,
            290
        )
    )

    draw_text(
        "SWEEP RESULTS",
        60,
        475,
        YELLOW,
        medium_font
    )

    for i, result in enumerate(
        sweep_results
    ):

        if name == "CG Height":

            value = (
                f"{result['value']:.2f} m"
            )

        elif name == "Mass":

            value = (
                f"{result['value']:.0f} kg"
            )

        elif name == "Front Weight Distribution":

            value = (
                f"{result['value'] * 100:.0f}%"
            )

        elif "C-alpha" in name:

            value = (
                f"{result['value']:.0f}"
            )

        else:

            value = (
                f"{result['value']:.2f}"
            )

        column = i % 7
        row = i // 7

        x = (
            70 +
            column * 175
        )

        y = (
            520 +
            row * 45
        )

        draw_text(
            value,
            x,
            y,
            WHITE,
            small_font
        )

        draw_text(
            f"{result['peak_g']:.2f} G",
            x,
            y + 18,
            GREEN,
            small_font
        )


# ============================================================
# SUSPENSION LAB
# ============================================================

suspension_parameters = [

    "Front Spring",
    "Rear Spring",
    "Front Damper",
    "Rear Damper",
    "Front Motion Ratio",
    "Rear Motion Ratio",
    "Front Unsprung Mass",
    "Rear Unsprung Mass",
    "Road Bump",
    "Front Roll Centre",
    "Rear Roll Centre",
    "Front ARB",
    "Rear ARB"
]

suspension_selected = 0


def suspension_value(name):

    if name == "Front Spring":

        return (
            f"{params.front_spring_rate / 1000:.1f} N/mm"
        )

    if name == "Rear Spring":

        return (
            f"{params.rear_spring_rate / 1000:.1f} N/mm"
        )

    if name == "Front Damper":

        return (
            f"{params.front_damper:.0f} Ns/m"
        )

    if name == "Rear Damper":

        return (
            f"{params.rear_damper:.0f} Ns/m"
        )

    if name == "Front Motion Ratio":

        return (
            f"{params.front_motion_ratio:.2f}"
        )

    if name == "Rear Motion Ratio":

        return (
            f"{params.rear_motion_ratio:.2f}"
        )

    if name == "Front Unsprung Mass":

        return (
            f"{params.front_unsprung_mass:.1f} kg"
        )

    if name == "Rear Unsprung Mass":

        return (
            f"{params.rear_unsprung_mass:.1f} kg"
        )

    if name == "Road Bump":

        return (
            f"{params.road_bump_height * 1000:.0f} mm"
        )

    if name == "Front Roll Centre":

        return (
            f"{params.front_roll_centre * 1000:.0f} mm"
        )

    if name == "Rear Roll Centre":

        return (
            f"{params.rear_roll_centre * 1000:.0f} mm"
        )

    if name == "Front ARB":

        return (
            f"{params.front_arb_stiffness:.0f} N*m/rad"
        )

    if name == "Rear ARB":

        return (
            f"{params.rear_arb_stiffness:.0f} N*m/rad"
        )

    return ""


def modify_suspension(
    name,
    direction
):

    if name == "Front Spring":

        params.front_spring_rate += (
            2000 *
            direction
        )

        params.front_spring_rate = max(
            10000,
            min(
                60000,
                params.front_spring_rate
            )
        )

    elif name == "Rear Spring":

        params.rear_spring_rate += (
            2000 *
            direction
        )

        params.rear_spring_rate = max(
            10000,
            min(
                60000,
                params.rear_spring_rate
            )
        )

    elif name == "Front Damper":

        params.front_damper += (
            200 *
            direction
        )

        params.front_damper = max(
            500,
            min(
                6000,
                params.front_damper
            )
        )

    elif name == "Rear Damper":

        params.rear_damper += (
            200 *
            direction
        )

        params.rear_damper = max(
            500,
            min(
                6000,
                params.rear_damper
            )
        )

    elif name == "Front Motion Ratio":

        params.front_motion_ratio += (
            0.05 *
            direction
        )

        params.front_motion_ratio = max(
            0.60,
            min(
                1.20,
                params.front_motion_ratio
            )
        )

    elif name == "Rear Motion Ratio":

        params.rear_motion_ratio += (
            0.05 *
            direction
        )

        params.rear_motion_ratio = max(
            0.60,
            min(
                1.20,
                params.rear_motion_ratio
            )
        )

    elif name == "Front Unsprung Mass":

        params.front_unsprung_mass += (
            1 *
            direction
        )

        params.front_unsprung_mass = max(
            10,
            min(
                40,
                params.front_unsprung_mass
            )
        )

    elif name == "Rear Unsprung Mass":

        params.rear_unsprung_mass += (
            1 *
            direction
        )

        params.rear_unsprung_mass = max(
            10,
            min(
                40,
                params.rear_unsprung_mass
            )
        )

    elif name == "Road Bump":

        params.road_bump_height += (
            0.005 *
            direction
        )

        params.road_bump_height = max(
            0.005,
            min(
                0.060,
                params.road_bump_height
            )
        )

    elif name == "Front Roll Centre":

        params.front_roll_centre += (
            0.005 *
            direction
        )

        params.front_roll_centre = max(
            0.0,
            min(
                0.100,
                params.front_roll_centre
            )
        )

    elif name == "Rear Roll Centre":

        params.rear_roll_centre += (
            0.005 *
            direction
        )

        params.rear_roll_centre = max(
            0.0,
            min(
                0.100,
                params.rear_roll_centre
            )
        )

    elif name == "Front ARB":

        params.front_arb_stiffness += (
            100.0 *
            direction
        )

        params.front_arb_stiffness = max(
            0.0,
            min(
                3000.0,
                params.front_arb_stiffness
            )
        )

    elif name == "Rear ARB":

        params.rear_arb_stiffness += (
            100.0 *
            direction
        )

        params.rear_arb_stiffness = max(
            0.0,
            min(
                3000.0,
                params.rear_arb_stiffness
            )
        )

    vehicle.create_suspension()


def draw_suspension_lab():

    screen.fill(BG)

    draw_text(
        "SUSPENSION + ROLL DYNAMICS LAB",
        35,
        25,
        WHITE,
        big_font
    )

    draw_text(
        "V0.4.3  |  ROLL CENTRE • ARB • LOAD TRANSFER",
        38,
        65,
        CYAN,
        small_font
    )

    panel(
        (
            35,
            110,
            650,
            650
        )
    )

    draw_text(
        "TUNING PARAMETERS",
        60,
        135,
        GREEN,
        medium_font
    )

    for i, name in enumerate(
        suspension_parameters
    ):

        selected = (
            i ==
            suspension_selected
        )

        y = (
            175 +
            i * 43
        )

        draw_text(
            ">"
            if selected
            else " ",
            55,
            y,
            GREEN,
            medium_font
        )

        draw_text(
            name,
            90,
            y,
            WHITE if selected else GREY,
            small_font
        )

        draw_text(
            suspension_value(name),
            410,
            y,
            YELLOW if selected else WHITE,
            small_font
        )

    draw_text(
        "↑ ↓ Select   ← → Adjust",
        60,
        690,
        CYAN,
        small_font
    )

    draw_text(
        "ENTER Drive   R Reset   ESC Back",
        60,
        720,
        CYAN,
        small_font
    )

    panel(
        (
            710,
            110,
            655,
            650
        )
    )

    draw_text(
        "LIVE ROLL DYNAMICS",
        740,
        135,
        YELLOW,
        medium_font
    )

    rows = [

        (
            "Lateral G",
            f"{vehicle.ay / 9.81:+.3f} G"
        ),

        (
            "Roll Angle",
            f"{math.degrees(vehicle.roll_angle):+.3f} deg"
        ),

        (
            "Roll Moment",
            f"{getattr(vehicle, 'roll_moment', 0.0):+.0f} Nm"
        ),

        (
            "Effective Roll Centre",
            f"{getattr(vehicle, 'effective_roll_centre', 0.0) * 1000:.1f} mm"
        ),

        (
            "Roll Arm",
            f"{getattr(vehicle, 'roll_arm', 0.0) * 1000:.1f} mm"
        ),

        (
            "Front Roll Stiffness",
            f"{getattr(vehicle, 'front_roll_stiffness_total', 0.0):.0f} Nm/rad"
        ),

        (
            "Rear Roll Stiffness",
            f"{getattr(vehicle, 'rear_roll_stiffness_total', 0.0):.0f} Nm/rad"
        ),

        (
            "Front Distribution",
            f"{getattr(vehicle, 'roll_stiffness_distribution_front', 0.0) * 100:.1f}%"
        ),

        (
            "Geometric F / R",
            f"{getattr(vehicle, 'geometric_transfer_front', 0.0):.0f} / "
            f"{getattr(vehicle, 'geometric_transfer_rear', 0.0):.0f} N"
        ),

        (
            "Elastic F / R",
            f"{getattr(vehicle, 'elastic_transfer_front', 0.0):.0f} / "
            f"{getattr(vehicle, 'elastic_transfer_rear', 0.0):.0f} N"
        )
    ]

    y = 180

    for label, value in rows:

        draw_text(
            label,
            740,
            y,
            GREY,
            small_font
        )

        draw_text(
            value,
            1010,
            y,
            WHITE,
            small_font
        )

        y += 32

    panel(
        (
            735,
            525,
            575,
            190
        ),
        PANEL2
    )

    draw_text(
        "PHYSICS CHAIN",
        760,
        545,
        CYAN,
        font
    )

    explanation = [

        "Lateral acceleration → CG roll moment",

        "Roll centre → geometric load transfer",

        "Springs + ARB → elastic load transfer",

        "F/R stiffness distribution → tyre load distribution",

        "Tyre load distribution → handling balance"
    ]

    y = 575

    for line in explanation:

        draw_text(
            line,
            760,
            y,
            WHITE,
            small_font
        )

        y += 25


# ============================================================
# TRACK SIMULATOR  -  V0.5
# ============================================================
# Architecture:
#   TrackPoint  - a single point on the centerline with direction
#   Track       - the closed Formula Student circuit
#   TrackSimulator - the full simulator: vehicle state, lap/sector
#                    timing, boundary detection, telemetry, drawing
# ============================================================

import colorsys


# ------------------------------------------------------------
# TRACK GEOMETRY
# ------------------------------------------------------------

class TrackPoint:
    """A single point on the track centerline."""

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def as_tuple(self):
        return (self.x, self.y)


class Track:
    """
    Closed Formula Student-style circuit built from Bezier control
    points.  The track is stored as a dense list of TrackPoints on
    the centerline together with the half-width at each sample.

    Sectors are defined by normalised progress thresholds:
      Sector 1 : 0.00 -> 0.33
      Sector 2 : 0.33 -> 0.66
      Sector 3 : 0.66 -> 1.00

    The start/finish line sits at progress = 0.0 / 1.0.
    """

    TRACK_WIDTH = 44        # half-width in pixels for drawing
    BOUNDARY_WIDTH = 50     # half-width for boundary detection
    SAMPLES = 1000          # resolution of the centerline polygon

    # Sector boundaries (normalised 0-1 progress)
    SECTOR_SPLITS = [0.33, 0.66]

    def __init__(self, cx, cy, scale=1.0):
        """
        Build the circuit.  cx/cy are the screen-space centre of the
        layout; scale allows the track to be zoomed.
        """
        self.cx = cx
        self.cy = cy
        self.scale = scale
        self._build()

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def _bezier_cubic(self, p0, p1, p2, p3, t):
        """Return a point on a cubic Bezier curve at parameter t."""
        mt = 1.0 - t
        x = (mt**3*p0[0] + 3*mt**2*t*p1[0] +
             3*mt*t**2*p2[0] + t**3*p3[0])
        y = (mt**3*p0[1] + 3*mt**2*t*p1[1] +
             3*mt*t**2*p2[1] + t**3*p3[1])
        return (x, y)

    def _build(self):
        """
        Define the track as a series of cubic Bezier segments.
        Coordinates are relative to (0,0); they are scaled and
        translated to screen space at the end.

        Track layout (full closed Formula Student-style circuit):
          S0  Start/Finish straight  (long, heading right)
          S1  Sweeping right-hander at corner 1
          S2  Short linking straight downward
          S3  Tight right hairpin at bottom-right
          S4  Long back straight heading left
          S5  Left chicane entry
          S6  Chicane exit / double-apex right
          S7  Medium straight heading up-left
          S8  Fast left sweeper back toward start side
          S9  Technical S-bend (right then left)
          S10 Return straight to S/F line
        """

        # Raw control-point segments  (each = 4 pts defining one cubic Bezier)
        # All coordinates in a normalised space (~-20 to +340 x, -60 to +290 y).
        # The circuit CLOSES perfectly at (0, 0) with a rightward heading.
        raw_segments = [
            # S0  Start/Finish long straight  (heading right)
            [(  0,   0), ( 80,   0), (160,   0), (250,   0)],
            # S1  Sweeping right-hander (top-right)
            [(250,   0), (310,   0), (330,  40), (330,  90)],
            # S2  Short straight curving downward
            [(330,  90), (330, 140), (315, 165), (290, 180)],
            # S3  Tight right hairpin at bottom-right
            [(290, 180), (270, 195), (260, 215), (270, 235)],
            [(270, 235), (280, 255), (305, 265), (320, 255)],
            # S4  Long back straight heading left
            [(320, 255), (325, 270), (310, 285), (280, 285)],
            [(280, 285), (220, 285), (140, 280), ( 60, 275)],
            # S5  Left chicane entry (bottom-left)
            [( 60, 275), ( 30, 272), (  5, 258), ( -5, 235)],
            # S6  Tight left hairpin / turnaround
            [( -5, 235), (-15, 215), ( -5, 195), ( 15, 190)],
            [(  15, 190), ( 35, 185), ( 55, 195), ( 60, 215)],
            # S7  Medium straight heading up
            [( 60, 215), ( 65, 240), ( 70, 220), ( 80, 195)],
            [( 80, 195), ( 90, 165), (100, 140), (110, 110)],
            # S8  Fast left sweeper heading toward top-left
            [(110, 110), (120,  80), (110,  45), ( 90,  20)],
            # S9  S-bend (right then left)
            [( 90,  20), ( 75,   5), ( 45,  -5), ( 20, -15)],
            [(  20, -15), (  5, -22), (-10, -18), (-15,  -8)],
            # S10  Return to start/finish  (tight left back onto main straight)
            [(-15,  -8), (-18,   0), (-10,   4), (  0,   0)],
        ]

        # Sample each Bezier segment
        raw_points = []
        n_per_seg = max(8, self.SAMPLES // len(raw_segments))
        for seg in raw_segments:
            p0, p1, p2, p3 = seg
            for i in range(n_per_seg):
                t = i / n_per_seg
                pt = self._bezier_cubic(p0, p1, p2, p3, t)
                raw_points.append(pt)

        # Scale and translate into screen space
        s = self.scale
        self.centerline = [
            TrackPoint(
                self.cx + pt[0] * s,
                self.cy + pt[1] * s
            )
            for pt in raw_points
        ]
        self.n = len(self.centerline)

        # Pre-compute cumulative arc-length for progress calculation
        self._arc = [0.0] * self.n
        total = 0.0
        for i in range(1, self.n):
            dx = self.centerline[i].x - self.centerline[i-1].x
            dy = self.centerline[i].y - self.centerline[i-1].y
            total += math.sqrt(dx*dx + dy*dy)
            self._arc[i] = total
        self.total_length = max(1.0, total)

        # Sector boundary indices
        self.sector_indices = []
        for thresh in self.SECTOR_SPLITS:
            target = thresh * self.total_length
            idx = 0
            for i, arc in enumerate(self._arc):
                if arc >= target:
                    idx = i
                    break
            self.sector_indices.append(idx)

        # Start/Finish position (index 0 on the centerline)
        self.start_pos = (
            self.centerline[0].x,
            self.centerline[0].y
        )

    # --------------------------------------------------------
    # QUERY HELPERS
    # --------------------------------------------------------

    def closest_index(self, x, y):
        """Return the index of the closest centerline point to (x,y)."""
        best_dist = float('inf')
        best_idx = 0
        # Optimisation: scan every 4th point then refine
        for i in range(0, self.n, 4):
            pt = self.centerline[i]
            d = (pt.x - x)**2 + (pt.y - y)**2
            if d < best_dist:
                best_dist = d
                best_idx = i
        # Refine in a window of ±8
        lo = max(0, best_idx - 8)
        hi = min(self.n - 1, best_idx + 8)
        for i in range(lo, hi + 1):
            pt = self.centerline[i]
            d = (pt.x - x)**2 + (pt.y - y)**2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, math.sqrt(best_dist)

    def progress_at(self, idx):
        """Return normalised progress [0,1) at a given centerline index."""
        return self._arc[idx] / self.total_length

    def sector_at(self, progress):
        """Return sector number 1, 2, or 3 for a given progress [0,1)."""
        if progress < self.SECTOR_SPLITS[0]:
            return 1
        elif progress < self.SECTOR_SPLITS[1]:
            return 2
        else:
            return 3

    def lateral_error(self, x, y, idx):
        """
        Signed lateral distance from the centerline at the closest
        point.  Positive = right of the track direction.
        """
        pt = self.centerline[idx]
        next_idx = (idx + 1) % self.n
        npt = self.centerline[next_idx]
        tx = npt.x - pt.x
        ty = npt.y - pt.y
        length = math.sqrt(tx*tx + ty*ty)
        if length < 1e-6:
            return 0.0
        # Normal pointing right of travel direction
        nx = ty / length
        ny = -tx / length
        dx = x - pt.x
        dy = y - pt.y
        return dx * nx + dy * ny

    # --------------------------------------------------------
    # DRAWING
    # --------------------------------------------------------

    def _centerline_pts(self):
        return [p.as_tuple() for p in self.centerline]

    def _offset_poly(self, side):
        """
        Compute the boundary polygon on the given side (+1 = right, -1 = left).
        """
        pts = []
        n = self.n
        for i in range(n):
            pt = self.centerline[i]
            nxt = self.centerline[(i + 1) % n]
            tx = nxt.x - pt.x
            ty = nxt.y - pt.y
            length = math.sqrt(tx*tx + ty*ty)
            if length < 1e-6:
                pts.append((pt.x, pt.y))
                continue
            nx = ty / length
            ny = -tx / length
            w = self.TRACK_WIDTH
            pts.append((
                pt.x + side * nx * w,
                pt.y + side * ny * w
            ))
        return pts

    @staticmethod
    def _label_pill(surface, text, font_obj, cx, cy, text_col, bg_col,
                    padding_x=7, padding_y=3):
        """Draw a text label with a rounded pill background, centred at (cx,cy)."""
        txt_surf = font_obj.render(text, True, text_col)
        tw, th = txt_surf.get_size()
        rx = cx - tw // 2 - padding_x
        ry = cy - th // 2 - padding_y
        rw = tw + padding_x * 2
        rh = th + padding_y * 2
        # shadow
        shadow_surf = pygame.Surface((rw + 2, rh + 2), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 0))
        pygame.draw.rect(shadow_surf, (0, 0, 0, 120),
                         (0, 0, rw + 2, rh + 2), border_radius=6)
        surface.blit(shadow_surf, (rx - 1, ry - 1))
        # pill
        pygame.draw.rect(surface, bg_col, (rx, ry, rw, rh), border_radius=5)
        surface.blit(txt_surf, (rx + padding_x, ry + padding_y))

    def draw(self, surface):
        """Draw the full track onto the surface."""
        cl_pts = self._centerline_pts()
        if len(cl_pts) < 2:
            return

        right_pts = self._offset_poly(+1)
        left_pts  = self._offset_poly(-1)

        # ── Track base surface (dark asphalt with subtle gradient effect) ──
        track_poly = right_pts + list(reversed(left_pts))
        if len(track_poly) >= 3:
            pygame.draw.polygon(surface, (28, 33, 42), track_poly)

        # ── Subtle track-edge glow (slightly lighter rim before the white line) ──
        if len(right_pts) >= 2:
            pygame.draw.lines(surface, (52, 60, 75), True, right_pts, 6)
            pygame.draw.lines(surface, (200, 210, 225), True, right_pts, 2)
        if len(left_pts) >= 2:
            pygame.draw.lines(surface, (52, 60, 75), True, left_pts, 6)
            pygame.draw.lines(surface, (200, 210, 225), True, left_pts, 2)

        # ── Kerb stripes (red/white alternating, every ~12 track-steps) ──
        kerb_step = max(1, self.n // 80)
        for i in range(0, self.n, kerb_step):
            is_red = (i // kerb_step) % 2 == 0
            color  = (210, 55, 55) if is_red else (230, 230, 230)
            pt  = self.centerline[i]
            nxt = self.centerline[(i + 1) % self.n]
            tx  = nxt.x - pt.x
            ty  = nxt.y - pt.y
            ln  = math.sqrt(tx * tx + ty * ty)
            if ln < 1e-6:
                continue
            nx = ty / ln
            ny = -tx / ln
            w  = self.TRACK_WIDTH
            # outer kerb
            pygame.draw.line(
                surface, color,
                (int(pt.x + nx * w),       int(pt.y + ny * w)),
                (int(pt.x + nx * (w + 7)), int(pt.y + ny * (w + 7))),
                4
            )
            # inner kerb
            pygame.draw.line(
                surface, color,
                (int(pt.x - nx * w),       int(pt.y - ny * w)),
                (int(pt.x - nx * (w + 7)), int(pt.y - ny * (w + 7))),
                4
            )

        # ── Centerline (dashed, muted cyan) ──
        dash_step = max(1, self.n // 100)
        for i in range(0, self.n - dash_step, dash_step * 2):
            x1 = int(self.centerline[i].x)
            y1 = int(self.centerline[i].y)
            x2 = int(self.centerline[i + dash_step].x)
            y2 = int(self.centerline[i + dash_step].y)
            pygame.draw.line(surface, (45, 130, 150), (x1, y1), (x2, y2), 1)

        # ── Sector boundary lines + pill labels ──
        sec_colors = [YELLOW, ORANGE, PURPLE]
        sec_bg     = [(60, 52, 10), (60, 38, 10), (40, 20, 70)]
        for sec_idx, idx in enumerate(self.sector_indices):
            pt  = self.centerline[idx]
            nxt = self.centerline[(idx + 1) % self.n]
            tx  = nxt.x - pt.x
            ty  = nxt.y - pt.y
            ln  = math.sqrt(tx * tx + ty * ty)
            if ln < 1e-6:
                continue
            nx = ty / ln
            ny = -tx / ln
            w   = self.TRACK_WIDTH
            col = sec_colors[sec_idx % 3]
            # full-width sector line (with glow)
            sx1 = int(pt.x - nx * w)
            sy1 = int(pt.y - ny * w)
            sx2 = int(pt.x + nx * w)
            sy2 = int(pt.y + ny * w)
            # glow (slightly dimmed wide line behind the sector line)
            glow_col = tuple(max(0, c - 60) for c in col)
            pygame.draw.line(surface, glow_col, (sx1, sy1), (sx2, sy2), 6)
            pygame.draw.line(surface, col, (sx1, sy1), (sx2, sy2), 2)
            # pill label well outside the outer boundary (65px past half-width)
            LABEL_OFFSET = w + 65
            lx = int(pt.x + nx * LABEL_OFFSET)
            ly = int(pt.y + ny * LABEL_OFFSET)
            self._label_pill(
                surface, f" S{sec_idx + 2} ",
                small_font, lx, ly,
                col, sec_bg[sec_idx % 3]
            )

        # ── Start/Finish line (chequered pattern across full track width) ──
        sf_pt  = self.centerline[0]
        sf_nxt = self.centerline[min(1, self.n - 1)]
        tx = sf_nxt.x - sf_pt.x
        ty = sf_nxt.y - sf_pt.y
        ln = math.sqrt(tx * tx + ty * ty)
        if ln > 1e-6:
            nx = ty / ln
            ny = -tx / ln
            w  = self.TRACK_WIDTH
            # chequered: 6 stripes across track width
            n_squares = 6
            for k in range(n_squares):
                frac0 = k / n_squares
                frac1 = (k + 1) / n_squares
                col   = (235, 235, 235) if k % 2 == 0 else (15, 15, 15)
                x1 = int(sf_pt.x - nx * w + nx * 2 * w * frac0)
                y1 = int(sf_pt.y - ny * w + ny * 2 * w * frac0)
                x2 = int(sf_pt.x - nx * w + nx * 2 * w * frac1)
                y2 = int(sf_pt.y - ny * w + ny * 2 * w * frac1)
                pygame.draw.line(surface, col, (x1, y1), (x2, y2), 6)
            # S/F pill label — well OUTSIDE the outer boundary
            LABEL_OFFSET = w + 65
            lx = int(sf_pt.x + nx * LABEL_OFFSET)
            ly = int(sf_pt.y + ny * LABEL_OFFSET)
            self._label_pill(
                surface, " S/F ",
                small_font, lx, ly,
                (30, 240, 130), (10, 60, 30)
            )


# ------------------------------------------------------------
# TRACK VEHICLE STATE  (bicycle model, uses existing params)
# ------------------------------------------------------------

class TrackVehicleState:
    """
    A separate, self-contained vehicle state for the Track Simulator.
    Uses the SAME VehicleParameters (params) object as the rest of the
    simulator so that Design Lab changes propagate automatically.

    Physics model: kinematic bicycle model with enhanced handling.
      yaw_rate = speed / wheelbase * tan(steering_angle)
      x       += speed * cos(heading) * dt
      y       += speed * sin(heading) * dt
      heading += yaw_rate * dt

    Acceleration/deceleration uses params.engine_acceleration,
    params.brake_deceleration, params.drag_coefficient and
    params.max_speed so changing the Design Lab values affects
    the simulator.

    Lateral g is derived from speed and yaw_rate.
    Load transfer modifies effective friction (using params.cg_height,
    params.mass, params.track).
    """

    # Maximum steering angle the driver can apply (radians)
    MAX_STEER = math.radians(28.0)

    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self, start_x=0.0, start_y=0.0, start_heading=0.0):
        self.x = start_x
        self.y = start_y
        self.heading = start_heading   # radians, screen coords
        self.speed = 0.0               # m/s (physics scale)
        self.steering = 0.0            # radians
        self.yaw_rate = 0.0            # rad/s
        self.throttle = 0.0
        self.brake_input = 0.0
        # Derived telemetry
        self.lateral_accel = 0.0       # m/s²
        self.long_accel = 0.0          # m/s²
        self.prev_speed = 0.0
        # Dynamic tyre loads (Newtons)
        self.fl_load = 612.5
        self.fr_load = 612.5
        self.rl_load = 612.5
        self.rr_load = 612.5

    def update(self, dt, throttle, brake, steer_dir):
        """
        Update the vehicle state for one time step.

        throttle : 0.0-1.0  (W key)
        brake    : 0.0-1.0  (S key)
        steer_dir: -1, 0, +1 (A/D keys)
        """
        p = self.p

        # -- Steering (rate-limited) --
        target_steer = steer_dir * self.MAX_STEER
        steer_rate = math.radians(120.0)   # deg/s responsiveness
        delta = target_steer - self.steering
        self.steering += math.copysign(
            min(abs(delta), steer_rate * dt), delta
        )

        # -- Effective friction (load-transfer aware) --
        # Higher lateral g or CG height reduces available friction
        lat_g = abs(self.lateral_accel) / 9.81
        mu_base = (p.front_mu + p.rear_mu) * 0.5
        # Simple load-transfer penalty: reduce mu at high lateral loads
        mu_effective = mu_base * max(0.6, 1.0 - 0.15 * lat_g)

        # -- Maximum possible cornering speed --
        # v_max_corner = sqrt(mu * g * R)  where R = wheelbase / tan(steer)
        # Used to automatically limit speed in corners (like a driver
        # modulating throttle to stay on the limit).
        min_steer_abs = max(0.01, abs(self.steering))
        corner_radius = p.wheelbase / math.tan(min_steer_abs)
        v_max_corner = math.sqrt(
            max(0.0, mu_effective * 9.81 * corner_radius)
        )
        v_max = min(p.max_speed, v_max_corner)

        # -- Longitudinal acceleration --
        # Drive force (from engine)
        drive_accel = throttle * p.engine_acceleration
        # Brake deceleration
        brake_accel = brake * p.brake_deceleration
        # Aerodynamic drag (proportional to v²)
        drag_accel = p.drag_coefficient * self.speed * abs(self.speed) / p.mass

        self.throttle = throttle
        self.brake_input = brake

        self.prev_speed = self.speed
        net_accel = drive_accel - brake_accel - drag_accel

        self.speed += net_accel * dt
        self.speed = max(0.0, min(p.max_speed, self.speed))

        # -- Yaw dynamics (kinematic bicycle model) --
        # yaw_rate = v / L * tan(delta)
        if self.speed > 0.1:
            self.yaw_rate = (
                self.speed /
                max(0.1, p.wheelbase) *
                math.tan(self.steering)
            )
        else:
            self.yaw_rate *= 0.85  # decay at standstill

        self.heading += self.yaw_rate * dt

        # -- Position update (screen coords: y increases downward) --
        # Scale from physics metres to screen pixels
        PIXELS_PER_METRE = 50.0
        self.x += self.speed * math.cos(self.heading) * dt * PIXELS_PER_METRE
        self.y += self.speed * math.sin(self.heading) * dt * PIXELS_PER_METRE

        # -- Derived telemetry --
        self.long_accel = (self.speed - self.prev_speed) / max(dt, 1e-6)
        # Lateral acceleration: centripetal = v * yaw_rate
        self.lateral_accel = self.speed * self.yaw_rate

        # -- 4-Tyre dynamic load transfer --
        m = p.mass
        g = 9.81
        w_total = m * g
        f_front_static = w_total * p.front_weight_distribution
        f_rear_static  = w_total * (1.0 - p.front_weight_distribution)

        # Longitudinal load transfer (accel transfers weight rearward, braking forward)
        delta_long = (m * self.long_accel * p.cg_height) / max(0.5, p.wheelbase)

        # Lateral load transfer (cornering transfers weight to outside wheels)
        delta_lat_front = (m * self.lateral_accel * p.cg_height * p.front_weight_distribution) / max(0.5, p.track)
        delta_lat_rear  = (m * self.lateral_accel * p.cg_height * (1.0 - p.front_weight_distribution)) / max(0.5, p.track)

        # Minimum tyre contact load
        min_load = 25.0
        self.fl_load = max(min_load, (f_front_static - delta_long) * 0.5 + delta_lat_front)
        self.fr_load = max(min_load, (f_front_static - delta_long) * 0.5 - delta_lat_front)
        self.rl_load = max(min_load, (f_rear_static  + delta_long) * 0.5 + delta_lat_rear)
        self.rr_load = max(min_load, (f_rear_static  + delta_long) * 0.5 - delta_lat_rear)


# ------------------------------------------------------------
# LAP TIMER
# ------------------------------------------------------------

class LapTimer:
    """
    Tracks lap count, current lap time, best lap time, sector times.
    Progress is 0.0-1.0 around the circuit.  A lap is completed when
    progress crosses 0.0 from above while having visited meaningful
    portions of the track (guarded by min_progress_required).
    """

    MIN_PROGRESS_BEFORE_LAP = 0.80   # must complete 80% before S/F counts

    def __init__(self, sector_splits):
        self.sector_splits = sector_splits   # e.g. [0.33, 0.66]
        self.reset()

    def reset(self):
        self.lap_number = 1
        self.lap_start_time = 0.0
        self.current_lap_time = 0.0
        self.best_lap_time = None
        self.elapsed = 0.0           # total running time
        self.prev_progress = 0.0
        self.max_progress_this_lap = 0.0
        # Sector state
        self.current_sector = 1
        self.sector_start_time = 0.0
        self.sector_times = [None, None, None]   # S1/S2/S3 for current lap
        self.best_sector_times = [None, None, None]
        self.last_sector_times = [None, None, None]

    def update(self, dt, progress):
        """
        Must be called every frame with the current normalised
        track progress [0, 1).
        """
        self.elapsed += dt
        self.current_lap_time = self.elapsed - self.lap_start_time

        # Track the maximum progress reached this lap (guards against
        # instant S/F crossing at spawn)
        if progress > self.max_progress_this_lap:
            self.max_progress_this_lap = progress

        prev = self.prev_progress
        cur  = progress

        # -- Sector crossing detection --
        for si, thresh in enumerate(self.sector_splits):
            sector_num = si + 1   # sectors are 1-indexed, this detects crossing INTO si+2
            if prev < thresh <= cur:
                # Crossed into the next sector
                t = self.current_lap_time
                self.sector_times[si] = t - self.sector_start_time
                self.last_sector_times[si] = self.sector_times[si]
                # Update best
                if (self.best_sector_times[si] is None or
                        self.sector_times[si] < self.best_sector_times[si]):
                    self.best_sector_times[si] = self.sector_times[si]
                self.sector_start_time = t
                self.current_sector = si + 2

        # -- Lap completion detection --
        # Detect downward crossing of the S/F line (progress wraps 1 -> 0)
        crossed_sf = (prev > self.MIN_PROGRESS_BEFORE_LAP and cur < 0.10)
        if crossed_sf and self.max_progress_this_lap > self.MIN_PROGRESS_BEFORE_LAP:
            lap_time = self.current_lap_time
            # Record final sector (S3)
            self.sector_times[2] = lap_time - self.sector_start_time
            self.last_sector_times[2] = self.sector_times[2]
            if (self.best_sector_times[2] is None or
                    self.sector_times[2] < self.best_sector_times[2]):
                self.best_sector_times[2] = self.sector_times[2]

            if self.best_lap_time is None or lap_time < self.best_lap_time:
                self.best_lap_time = lap_time

            # Start new lap
            self.lap_number += 1
            self.lap_start_time = self.elapsed
            self.current_lap_time = 0.0
            self.sector_times = [None, None, None]
            self.current_sector = 1
            self.sector_start_time = 0.0
            self.max_progress_this_lap = cur

        self.prev_progress = cur

    @staticmethod
    def format_time(t):
        """Format a time in seconds as MM:SS.mmm"""
        if t is None:
            return "--:--.---"
        minutes = int(t // 60)
        seconds = int(t % 60)
        millis  = int((t % 1) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


# ------------------------------------------------------------
# TRACK SIMULATOR
# ------------------------------------------------------------

# ============================================================
# SPEED HEATMAP
# ============================================================

class SpeedHeatmap:
    """
    Records the maximum speed seen at every centerline index during
    a lap and colors the track accordingly:
      slow  ->  deep blue
      mid   ->  yellow
      fast  ->  bright red

    The heatmap is rebuilt each time a new lap is completed so it
    always reflects your latest performance.
    """

    def __init__(self, n_points):
        self.n = n_points
        # Speed in km/h at each track index for the current lap
        self._current = [0.0] * n_points
        # Confirmed best (shown on track)
        self.best = [0.0] * n_points
        self._surface = None   # cached pygame Surface
        self._dirty = True     # needs redraw

    def update(self, idx, speed_kmh):
        """Record the maximum speed seen at this index this lap."""
        if speed_kmh > self._current[idx]:
            self._current[idx] = speed_kmh

    def commit_lap(self):
        """Called when a lap completes — save current as best."""
        for i in range(self.n):
            if self._current[i] > self.best[i]:
                self.best[i] = self._current[i]
        self._current = [0.0] * self.n
        self._dirty = True

    @staticmethod
    def _speed_color(speed_kmh, max_speed):
        """Map 0-max_speed to blue->cyan->green->yellow->red."""
        if max_speed < 1.0:
            return (60, 60, 200)
        t = max(0.0, min(1.0, speed_kmh / max_speed))
        # HSL: 240 deg (blue) at t=0  ->  0 deg (red) at t=1
        hue = (1.0 - t) * 0.667   # 0.667 = blue, 0.0 = red
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def build_surface(self, centerline, width_px=6):
        """Render the heatmap onto a transparent surface."""
        self._surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self._surface.fill((0, 0, 0, 0))
        if not any(v > 0 for v in self.best):
            self._dirty = False
            return
        max_spd = max(self.best)
        n = len(centerline)
        for i in range(n - 1):
            col = self._speed_color(self.best[i], max_spd)
            x1 = int(centerline[i].x)
            y1 = int(centerline[i].y)
            x2 = int(centerline[i+1].x)
            y2 = int(centerline[i+1].y)
            pygame.draw.line(self._surface, col, (x1, y1), (x2, y2),
                             width_px)
        self._dirty = False

    def draw(self, surface, centerline, cam_x, cam_y):
        """Blit the heatmap surface with camera offset."""
        if self._dirty or self._surface is None:
            self.build_surface(centerline)
        if self._surface is not None:
            surface.blit(self._surface, (-cam_x, -cam_y))


# ============================================================
# GHOST CAR
# ============================================================

class GhostCar:
    """
    Records the vehicle's (x, y, heading, speed) every frame during
    a lap.  When a new best lap is set the recording is promoted to
    the 'best ghost'.  During subsequent laps the ghost is replayed
    frame-by-frame as a translucent blue arrow so the driver can see
    exactly where time is gained or lost.
    """

    def __init__(self):
        self._recording = []   # [(x, y, heading, speed_kmh), ...]
        self._ghost     = []   # best lap recording
        self._frame     = 0    # playback index
        self.enabled    = True
        self.has_ghost  = False

    def reset_recording(self):
        """Start a fresh recording for the current lap."""
        self._recording = []
        self._frame = 0

    def record(self, x, y, heading, speed_kmh):
        """Save one frame."""
        self._recording.append((x, y, heading, speed_kmh))

    def promote(self):
        """Called when this lap is a new best — make it the ghost."""
        self._ghost = list(self._recording)
        self.has_ghost = bool(self._ghost)
        self._frame = 0

    def tick(self):
        """Advance the ghost playback by one frame."""
        if self._ghost:
            self._frame = (self._frame + 1) % len(self._ghost)

    @property
    def ghost_x(self):
        if self.has_ghost and self._ghost:
            idx = min(self._frame, len(self._ghost) - 1)
            return self._ghost[idx][0]
        return 0.0

    @property
    def ghost_y(self):
        if self.has_ghost and self._ghost:
            idx = min(self._frame, len(self._ghost) - 1)
            return self._ghost[idx][1]
        return 0.0

    @property
    def ghost_heading(self):
        if self.has_ghost and self._ghost:
            idx = min(self._frame, len(self._ghost) - 1)
            return self._ghost[idx][2]
        return 0.0

    def draw(self, surface, cam_x, cam_y):
        """Draw the ghost car at its current replay position."""
        if not self.enabled or not self.has_ghost or not self._ghost:
            return
        idx = min(self._frame, len(self._ghost) - 1)
        gx, gy, gh, gspd = self._ghost[idx]
        sx = gx - cam_x
        sy = gy - cam_y
        c = math.cos(gh)
        s = math.sin(gh)

        def tf(lx, ly):
            return (int(sx + lx*c - ly*s), int(sy + lx*s + ly*c))

        length, width = 18, 8
        body = [tf(length, 0), tf(-length//2, width), tf(-length//2, -width)]
        # Draw semi-transparent ghost using a temp surface
        ghost_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(ghost_surf, (80, 180, 255, 130), body)
        pygame.draw.polygon(ghost_surf, (180, 220, 255, 160), body, 1)
        cx_s, cy_s = tf(0, 0)
        pygame.draw.circle(ghost_surf, (200, 230, 255, 180), (cx_s, cy_s), 4)
        surface.blit(ghost_surf, (0, 0))
        # Speed delta label
        delta_spd = self.vehicle_speed - gspd if hasattr(self, 'vehicle_speed') else 0
        col = GREEN if delta_spd >= 0 else RED
        ds = small_font.render(f"{delta_spd:+.1f}", True, col)
        surface.blit(ds, (int(sx) + 10, int(sy) - 14))


# ============================================================
# CSV EXPORTER
# ============================================================

import csv
import os
import datetime

class LapDataExporter:
    """
    Collects per-frame telemetry during a lap and writes it to a
    CSV file when triggered.  The file is saved in the same folder
    as main.py so it's easy to open in Excel or Python/pandas.

    Columns:
      time_s, lap, progress, speed_kmh, throttle, brake,
      steering_deg, yaw_rate_dps, lat_g, long_g,
      lateral_error_m, sector
    """

    def __init__(self):
        self._rows = []
        self._lap_rows = []   # rows for current lap only

    def record(self, t, lap, progress, speed_kmh, throttle, brake,
               steer_deg, yaw_dps, lat_g, long_g, lat_err_m, sector):
        row = [round(t,4), lap, round(progress,4),
               round(speed_kmh,2), round(throttle,2), round(brake,2),
               round(steer_deg,2), round(yaw_dps,2),
               round(lat_g,4), round(long_g,4),
               round(lat_err_m,4), sector]
        self._lap_rows.append(row)

    def commit_lap(self):
        """Move lap rows into the main buffer when a lap completes."""
        self._rows.extend(self._lap_rows)
        self._lap_rows = []

    def export(self):
        """Write all collected data to a timestamped CSV file."""
        all_rows = self._rows + self._lap_rows
        if not all_rows:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"lap_telemetry_{ts}.csv"
        )
        headers = [
            "time_s", "lap", "progress", "speed_kmh",
            "throttle", "brake", "steering_deg", "yaw_rate_dps",
            "lat_g", "long_g", "lateral_error_m", "sector"
        ]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(all_rows)
        return filename


# ============================================================
# TRACK SIMULATOR  (upgraded)
# ============================================================

class TrackSimulator:
    """
    Full Track Simulator screen — V0.6

    New in V0.6:
      - Speed heatmap  (P to toggle)
      - Ghost car      (best lap replayed, G to toggle)
      - CSV export     (F key)
      - Live G-force friction circle
      - Delta timer vs best lap
    """

    LEFT_PANEL_W  = 280
    RIGHT_PANEL_W = 540  # wider to hold HUD overlays without track collision
    MAP_SIZE      = 200  # minimap square size

    # World-space track centre and scale
    # New layout raw bounds: x in [-20,330], y in [-30,290]
    # Midpoint ~(155,130).  Usable view centre (between panels): (~710,425).
    # => TRACK_CX = 710 - 155*2.2 ≈ 369,  TRACK_CY = 425 - 130*2.2 ≈ 139
    TRACK_CX    = 369
    TRACK_CY    = 139
    TRACK_SCALE = 2.2

    # Off-track detection threshold (pixels from centerline)
    OFF_TRACK_THRESHOLD = 50
    OFF_TRACK_SPEED_PENALTY = 0.6

    def __init__(self, p):
        self.p = p
        # Build the track
        self.track = Track(
            self.TRACK_CX, self.TRACK_CY, self.TRACK_SCALE
        )
        # Vehicle state
        self.vehicle = TrackVehicleState(p)
        # Lap timer
        self.lap_timer = LapTimer(self.track.SECTOR_SPLITS)
        # Spawn the vehicle at the start/finish line
        self._place_vehicle_at_start()
        # State flags
        self.off_track = False
        self.off_track_timer = 0.0
        self.max_speed_seen = 0.0
        # Camera offset (for scrolling track view)
        self.cam_x = 0.0
        self.cam_y = 0.0
        # Track progress
        self.progress = 0.0
        self.lateral_error = 0.0
        self.closest_idx = 0
        # Pre-rendered track surface (for performance)
        self._track_surface = None
        self._build_track_surface()
        # V0.6 systems
        self.heatmap = SpeedHeatmap(self.track.n)
        self.show_heatmap = True         # P to toggle
        self.ghost = GhostCar()
        self.ghost.enabled = True        # G to toggle
        self.exporter = LapDataExporter()
        self._elapsed = 0.0              # total elapsed time for CSV
        self._export_notice = 0.0        # seconds to show export notice
        self._export_filename = ""
        # Delta timer: interpolated best-lap time at current progress
        self._best_lap_splits = []       # (progress, elapsed) pairs from best lap
        self._recording_splits = []      # live: (progress, elapsed) for current lap
        self._lap_start_elapsed = 0.0
        # Stacking anchor for NuclearHUD panels (updated each frame by _draw_right_panel)
        self._right_panel_hud_y = HEIGHT // 2
        # ---- V2.0: Visual effects state ----
        # Motion blur trail: list of (x, y, heading, alpha) world-space positions
        self._motion_trail = []
        # Camera shake: trauma decays over time
        self._shake_trauma = 0.0
        self._cam_shake_dx = 0.0
        self._cam_shake_dy = 0.0
        # Exhaust heat shimmer particles: list of [wx, wy, vx, vy, life, max_life]
        self._exhaust_particles = []
        self._exhaust_timer = 0.0
        # Suspension visual state (smoothed)
        self._vis_roll  = 0.0   # body roll offset in car-local coords (pixels)
        self._vis_pitch = 0.0   # nose-up/-down pixel offset

        # ---- V3.0: 3D Real Driving View System ----
        # 0: 3D Chase Camera, 1: 3D Cockpit FPV, 2: 2D Tactical Top-Down
        self.camera_mode = 0
        self.wide_mode = False
        self.cam_3d_x = self.vehicle.x
        self.cam_3d_y = self.vehicle.y
        self.cam_3d_z = 25.0
        self.cam_3d_heading = self.vehicle.heading
        self.cam_3d_pitch = 0.0
        self.cam_3d_roll = 0.0
        self._gear = 1
        self._rpm = 2500.0
        self._flame_timer = 0.0
        self._prev_throttle = 0.0
        self._tire_smoke = []
        self._steer_wheel_visual = 0.0
        self._init_3d_world()

    def _place_vehicle_at_start(self):
        """Teleport the vehicle to the S/F line with correct heading."""
        cl = self.track.centerline
        sx = cl[0].x
        sy = cl[0].y
        # Heading: direction of first segment
        nx = cl[1].x - cl[0].x
        ny = cl[1].y - cl[0].y
        heading = math.atan2(ny, nx)
        self.vehicle.reset(sx, sy, heading)
        self.progress = 0.0
        self.lateral_error = 0.0
        self.closest_idx = 0

    def _build_track_surface(self):
        """Pre-render the static track onto an off-screen surface."""
        self._track_surface = pygame.Surface(
            (WIDTH, HEIGHT), pygame.SRCALPHA
        )
        self._track_surface.fill((0, 0, 0, 0))
        self.track.draw(self._track_surface)

    def reset_lap(self):
        """R key: reset position and lap timer but keep best lap."""
        best = self.lap_timer.best_lap_time
        best_sectors = list(self.lap_timer.best_sector_times)
        self.lap_timer.reset()
        self.lap_timer.best_lap_time = best
        self.lap_timer.best_sector_times = best_sectors
        self._place_vehicle_at_start()
        self.off_track = False
        self.off_track_timer = 0.0
        self.max_speed_seen = 0.0
        self.ghost.reset_recording()
        self._recording_splits = []
        self._lap_start_elapsed = self._elapsed
        self.exporter._lap_rows = []

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def _check_lap_completed(self):
        """
        Called after lap_timer.update().  If a new lap started
        (lap_number incremented), commit the ghost/heatmap/exporter
        and promote if it was a best lap.
        """
        pass   # Logic is handled inline in update() by watching lap_number.

    def update(self, dt, throttle, brake, steer_dir):
        """Called every frame while in TRACK mode."""
        self._elapsed += dt
        if self._export_notice > 0:
            self._export_notice -= dt

        # Update vehicle physics
        self.vehicle.update(dt, throttle, brake, steer_dir)

        # Find closest track point
        idx, dist_px = self.track.closest_index(
            self.vehicle.x, self.vehicle.y
        )
        self.closest_idx = idx
        self.progress = self.track.progress_at(idx)
        self.lateral_error = self.track.lateral_error(
            self.vehicle.x, self.vehicle.y, idx
        )

        # Off-track detection
        if dist_px > self.track.BOUNDARY_WIDTH:
            self.off_track = True
            self.off_track_timer += dt
            # Speed penalty when off-track
            self.vehicle.speed *= self.OFF_TRACK_SPEED_PENALTY ** dt
        else:
            self.off_track = False
            self.off_track_timer = 0.0

        spd_kmh = self.vehicle.speed * 3.6
        if spd_kmh > self.max_speed_seen:
            self.max_speed_seen = spd_kmh

        # ---- V0.6: heatmap update ----
        self.heatmap.update(idx, spd_kmh)

        # ---- V0.6: ghost recording ----
        self.ghost.vehicle_speed = spd_kmh
        self.ghost.record(
            self.vehicle.x, self.vehicle.y,
            self.vehicle.heading, spd_kmh
        )
        self.ghost.tick()

        # ---- V0.6: delta-timer split recording ----
        self._recording_splits.append(
            (self.progress,
             self._elapsed - self._lap_start_elapsed)
        )

        # ---- V0.6: CSV telemetry recording ----
        lat_g  = self.vehicle.lateral_accel / 9.81
        long_g = self.vehicle.long_accel    / 9.81
        lat_err_m = self.lateral_error / 50.0
        self.exporter.record(
            self._elapsed,
            self.lap_timer.lap_number,
            self.progress,
            spd_kmh,
            self.vehicle.throttle,
            self.vehicle.brake_input,
            math.degrees(self.vehicle.steering),
            math.degrees(self.vehicle.yaw_rate),
            lat_g, long_g,
            lat_err_m,
            self.lap_timer.current_sector
        )

        # ---- Lap completion: promote ghost / commit heatmap ----
        prev_lap = self.lap_timer.lap_number
        self.lap_timer.update(dt, self.progress)
        if self.lap_timer.lap_number != prev_lap:
            # A lap was completed.  Check if it was a best lap.
            prev_best = self.lap_timer.best_lap_time
            completed_time = self.lap_timer.current_lap_time  # 0 = just reset
            # 'best_lap_time' was already updated inside LapTimer.update
            # If the ghost recording matches the new best, promote it.
            self.ghost.promote()   # always promote last lap as candidate
            self.heatmap.commit_lap()
            self.heatmap._dirty = True
            self.exporter.commit_lap()
            # Store split data for delta timer
            self._best_lap_splits = list(self._recording_splits)
            self._recording_splits = []
            self._lap_start_elapsed = self._elapsed
            self.ghost.reset_recording()

        # Camera follows vehicle in 2D mode (centered in viewport)
        view_cx = self.LEFT_PANEL_W + (WIDTH - self.LEFT_PANEL_W - self.RIGHT_PANEL_W) * 0.5
        view_cy = HEIGHT * 0.5
        target_cam_x = self.vehicle.x - view_cx
        target_cam_y = self.vehicle.y - view_cy
        lag = 1.0 - math.exp(-16.0 * dt)
        self.cam_x += (target_cam_x - self.cam_x) * lag
        self.cam_y += (target_cam_y - self.cam_y) * lag

        # V3.0: Update 3D Camera & dynamic driving physics
        self._update_3d_physics(dt)

    # --------------------------------------------------------
    # DRAWING
    # --------------------------------------------------------

    def _world_to_screen(self, wx, wy):
        """Convert world pixel coords to screen coords (camera applied)."""
        return (wx - self.cam_x, wy - self.cam_y)

    def _draw_vehicle(self, surface):
        """
        Render a detailed 2.5D Formula Student car with:
          - Multi-polygon bodywork (nose, sidepods, cockpit, roll hoop)
          - Front and rear wings with endplates
          - Live tyre deformation (wider under lateral G, squish under braking)
          - Body roll / pitch via suspension telemetry
          - Ground shadow (soft blurred ellipse)
          - Exhaust heat shimmer glow at rear
        """
        veh = self.vehicle
        sx, sy = self._world_to_screen(veh.x, veh.y)
        # Apply camera shake offset
        sx += self._cam_shake_dx
        sy += self._cam_shake_dy

        h  = veh.heading
        ch = math.cos(h)
        sh = math.sin(h)

        lat_g  = veh.lateral_accel / 9.81
        long_g = veh.long_accel / 9.81
        speed  = veh.speed  # m/s

        # ---- Smoothed visual roll / pitch ----
        target_roll  = lat_g  * 2.5   # pixels of body roll
        target_pitch = -long_g * 1.8  # pixels of nose dive/lift
        SMOOTH = 0.18
        self._vis_roll  += (target_roll  - self._vis_roll)  * SMOOTH
        self._vis_pitch += (target_pitch - self._vis_pitch) * SMOOTH
        roll_offset  = self._vis_roll
        pitch_offset = self._vis_pitch

        # Local -> screen transform (with roll shift applied laterally)
        def tf(lx, ly):
            """Transform car-local coords to screen.
            lx = forward (+front), ly = right (+right)
            Roll is applied as a lateral shift to the body."""
            lx2 = lx + pitch_offset
            ly2 = ly + roll_offset
            return (
                int(sx + lx2 * ch - ly2 * sh),
                int(sy + lx2 * sh + ly2 * ch)
            )

        # ---- Tyre deformation parameters ----
        tyre_w_base = 5        # half-width of tyre contact patch (px)
        tyre_h_base = 8        # half-length of tyre contact patch (px)
        lat_g_abs   = abs(lat_g)
        long_g_abs  = abs(long_g)
        # Lateral G widens tyres, braking squishes them flatter
        front_tw = tyre_w_base + lat_g_abs * 2.2
        front_th = tyre_h_base - long_g_abs * 1.5
        rear_tw  = tyre_w_base + lat_g_abs * 2.8   # rear squirm more
        rear_th  = tyre_h_base - long_g_abs * 1.2
        front_tw = max(3.0, min(11.0, front_tw))
        front_th = max(4.0, min(12.0, front_th))
        rear_tw  = max(3.0, min(13.0, rear_tw))
        rear_th  = max(4.0, min(12.0, rear_th))

        # ---- Suspension bounce (wheel travel relative to body) ----
        # Approximate from lateral G / long G — bumpier at high speed on kerbs
        bounce_amp = 1.5 if self.off_track else 0.3
        t_bounce   = self._elapsed * 18.0
        fl_z = math.sin(t_bounce + 0.0) * bounce_amp * (1 + lat_g_abs * 0.4)
        fr_z = math.sin(t_bounce + 1.1) * bounce_amp * (1 - lat_g_abs * 0.4)
        rl_z = math.sin(t_bounce + 2.3) * bounce_amp
        rr_z = math.sin(t_bounce + 3.5) * bounce_amp

        # ========================================================
        # GROUND SHADOW  (soft ellipse, offset slightly forward)
        # ========================================================
        shadow_surf = pygame.Surface((80, 34), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 55), (0, 0, 80, 34))
        # Rotate the shadow surface to match heading
        rotated_shadow = pygame.transform.rotate(
            shadow_surf, -math.degrees(h)
        )
        srx = rotated_shadow.get_width()  // 2
        sry = rotated_shadow.get_height() // 2
        shadow_center = tf(0, 0)
        surface.blit(rotated_shadow,
                     (shadow_center[0] - srx + 4, shadow_center[1] - sry + 5))

        # ========================================================
        # EXHAUST GLOW  (rear diffuser area)
        # ========================================================
        exhaust_alpha = max(0, min(200, int(speed * 9)))
        if exhaust_alpha > 20:
            ex_cx, ex_cy = tf(-20, 0)
            glow_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
            # Multi-layer glow: outer (orange) -> inner (bright yellow)
            for r_glow, col_glow in [
                (18, (255, 100, 20, max(0, exhaust_alpha // 5))),
                (12, (255, 160, 40, max(0, exhaust_alpha // 3))),
                (6,  (255, 220, 80, max(0, exhaust_alpha // 2))),
            ]:
                pygame.draw.circle(glow_surf, col_glow, (20, 10), r_glow)
            rot_glow = pygame.transform.rotate(glow_surf, -math.degrees(h))
            surface.blit(rot_glow,
                         (ex_cx - rot_glow.get_width()  // 2,
                          ex_cy - rot_glow.get_height() // 2))

        # ========================================================
        # HEAT SHIMMER PARTICLES  (rising from exhaust)
        # ========================================================
        for px_p, py_p, pvx, pvy, plife, pmax in self._exhaust_particles:
            ratio = plife / max(pmax, 0.001)
            a = int(60 * ratio)
            if a > 2:
                try:
                    shimmer_s = pygame.Surface((6, 6), pygame.SRCALPHA)
                    pygame.draw.circle(shimmer_s, (255, 200, 80, a), (3, 3), 3)
                    sx_p, sy_p = self._world_to_screen(px_p, py_p)
                    surface.blit(shimmer_s, (int(sx_p) - 3, int(sy_p) - 3))
                except Exception:
                    pass

        # ========================================================
        # REAR WING ASSEMBLY
        # ========================================================
        # Main plane
        rw_pts = [tf(-19, -12), tf(-19, 12), tf(-17, 11), tf(-17, -11)]
        pygame.draw.polygon(surface, (55, 65, 80), rw_pts)
        pygame.draw.polygon(surface, (180, 190, 210), rw_pts, 1)
        # End plates
        for side in [-1, 1]:
            ep = [tf(-16, side*11), tf(-21, side*11),
                  tf(-21, side*13), tf(-16, side*13)]
            pygame.draw.polygon(surface, (45, 55, 70), ep)
            pygame.draw.polygon(surface, (150, 160, 180), ep, 1)
        # DRS indicator stripe (red when not active, green when active)
        drs_active = getattr(getattr(self, '_nuclear_aero', None),
                             'drs_active', False)
        drs_col = (30, 220, 80) if drs_active else (180, 50, 50)
        pygame.draw.line(surface, drs_col, tf(-17, -11), tf(-17, 11), 2)

        # ========================================================
        # MAIN BODY / MONOCOQUE
        # ========================================================
        # Underbody plate (slightly wider, darker)
        under = [tf(-19, -9), tf(12, -5), tf(14, 0), tf(12, 5), tf(-19, 9)]
        pygame.draw.polygon(surface, (22, 28, 38), under)

        # Sidepods (left and right)
        pod_col = (35, 42, 58)
        pod_hi  = (55, 68, 92)
        for side in [-1, 1]:
            pod = [
                tf( -15, side * 6),
                tf(   5, side * 7),
                tf(   8, side * 5),
                tf(  -2, side * 4),
            ]
            pygame.draw.polygon(surface, pod_col, pod)
            pygame.draw.polygon(surface, pod_hi,  pod, 1)

        # Central spine / tub
        spine = [tf(-16, -4), tf(13, -3), tf(15, 0), tf(13, 3), tf(-16, 4)]
        spine_col = (45, 55, 75)
        pygame.draw.polygon(surface, spine_col, spine)

        # ========================================================
        # NOSE CONE  (tapered, colour accent)
        # ========================================================
        nose_col   = (50, 135, 220)   # blue livery accent
        nose_hi    = (90, 175, 255)
        nose_dark  = (25, 75, 145)
        nose = [tf(13, -3), tf(15, 0), tf(13, 3), tf(21, 1), tf(21, -1)]
        pygame.draw.polygon(surface, nose_col, nose)
        pygame.draw.polygon(surface, nose_hi,  nose, 1)

        # ========================================================
        # COCKPIT SURROUND + HALO
        # ========================================================
        # Cockpit opening (dark recess)
        cockpit = [tf(2, -3.5), tf(6, -2), tf(6, 2), tf(2, 3.5),
                   tf(-4, 3), tf(-4, -3)]
        pygame.draw.polygon(surface, (12, 16, 22), cockpit)
        pygame.draw.polygon(surface, (60, 75, 100), cockpit, 1)
        # Halo safety structure (titanium colour)
        halo_col = (140, 150, 165)
        # Halo arch: two points to a peak point
        p_left  = tf(0, -3.5)
        p_right = tf(0,  3.5)
        p_peak  = tf(1,  0)
        pygame.draw.line(surface, halo_col, p_left,  p_peak, 3)
        pygame.draw.line(surface, halo_col, p_right, p_peak, 3)
        pygame.draw.line(surface, (160, 170, 185), p_left,  p_peak, 1)
        pygame.draw.line(surface, (160, 170, 185), p_right, p_peak, 1)
        # Roll hoop
        roll_top = tf(-5, 0)
        roll_l   = tf(-6, -3)
        roll_r   = tf(-6,  3)
        pygame.draw.polygon(surface, (90, 100, 120),
                            [roll_l, roll_top, roll_r])
        pygame.draw.polygon(surface, (160, 170, 195),
                            [roll_l, roll_top, roll_r], 1)
        # Driver helmet
        helm_cx, helm_cy = tf(3, 0)
        pygame.draw.circle(surface, (220, 80, 30), (helm_cx, helm_cy), 4)
        pygame.draw.circle(surface, (255, 130, 60), (helm_cx, helm_cy), 4, 1)
        # Visor slit
        vis_l = tf(3.5, -2)
        vis_r = tf(3.5,  2)
        pygame.draw.line(surface, (150, 220, 255), vis_l, vis_r, 1)

        # ========================================================
        # FRONT WING ASSEMBLY
        # ========================================================
        # Main front wing plane
        fw_pts = [tf(20, -14), tf(20, 14), tf(18, 13), tf(18, -13)]
        pygame.draw.polygon(surface, (40, 50, 68), fw_pts)
        pygame.draw.polygon(surface, (160, 170, 195), fw_pts, 1)
        # Cascade / secondary flap
        fw2 = [tf(21, -12), tf(21, 12), tf(20, 11), tf(20, -11)]
        pygame.draw.polygon(surface, (60, 75, 100), fw2)
        # Endplates
        for side in [-1, 1]:
            fep = [tf(19, side*13), tf(21, side*13),
                   tf(22, side*10), tf(19, side*12)]
            pygame.draw.polygon(surface, (45, 55, 72), fep)
            pygame.draw.polygon(surface, (130, 145, 170), fep, 1)

        # ========================================================
        # TYRES  (deformable contact patches)
        # ========================================================
        tyre_col     = (20, 20, 22)
        tyre_hi_col  = (60, 60, 65)
        tyre_hot_col = (80, 40, 20)  # colour when working hard
        tyre_heat    = min(1.0, (lat_g_abs + long_g_abs) * 0.4)

        wheel_positions = [
            # (local_x, local_y, tyre_w, tyre_h, z_bounce)
            ( 18,  -9, front_tw, front_th, fl_z),   # front-left (outer)
            ( 18,   9, front_tw, front_th, fr_z),   # front-right (outer)
            (-13,  -9, rear_tw,  rear_th,  rl_z),   # rear-left
            (-13,   9, rear_tw,  rear_th,  rr_z),   # rear-right
        ]
        for wlx, wly, tw, th, wz in wheel_positions:
            # Apply suspension bounce: z-offset translates to a vertical shift
            # in screen coords (perpendicular to heading direction)
            bx = wz * sh   # bounce shifts the wheel perpendicular to fwd
            by = -wz * ch
            wsx = int(sx + (wlx + pitch_offset) * ch - (wly + roll_offset) * sh + bx)
            wsy = int(sy + (wlx + pitch_offset) * sh + (wly + roll_offset) * ch + by)

            # Tyre sidewall (main oval)
            # We draw an ellipse-like polygon for the tyre patch
            n_pts = 12
            tyre_pts = []
            for k in range(n_pts):
                angle_k = 2 * math.pi * k / n_pts
                lx_t = math.cos(angle_k) * tw
                ly_t = math.sin(angle_k) * th
                # Rotate by car heading
                tx_t = lx_t * ch - ly_t * sh
                ty_t = lx_t * sh + ly_t * ch
                tyre_pts.append((int(wsx + tx_t), int(wsy + ty_t)))
            # Blend towards hot colour based on tyre_heat
            r = int(tyre_col[0] + (tyre_hot_col[0] - tyre_col[0]) * tyre_heat)
            g_c = int(tyre_col[1] + (tyre_hot_col[1] - tyre_col[1]) * tyre_heat)
            b_c = int(tyre_col[2] + (tyre_hot_col[2] - tyre_col[2]) * tyre_heat)
            pygame.draw.polygon(surface, (r, g_c, b_c), tyre_pts)
            pygame.draw.polygon(surface, tyre_hi_col, tyre_pts, 1)
            # Wheel rim (small inner circle)
            rim_col = (120, 125, 140)
            pygame.draw.circle(surface, rim_col, (wsx, wsy), max(2, int(tw * 0.55)))
            pygame.draw.circle(surface, (160, 165, 180), (wsx, wsy),
                               max(2, int(tw * 0.55)), 1)

        # ========================================================
        # LIVERY ACCENTS (colour stripe along spine)
        # ========================================================
        stripe_col = (50, 135, 220)
        pygame.draw.line(surface, stripe_col, tf(13, 0), tf(-16, 0), 2)
        # Rear diffuser highlight
        pygame.draw.line(surface, (255, 160, 40), tf(-16, -7), tf(-16, 7), 3)


    def _delta_at_progress(self, progress):
        """
        Interpolate what the best-lap elapsed time was at the given
        progress value.  Returns None if no best lap splits exist.
        Returns the delta in seconds: positive = ahead, negative = behind.
        """
        if not self._best_lap_splits or not self._recording_splits:
            return None
        # Find best-lap time at current progress
        best_t = None
        for bp, bt in self._best_lap_splits:
            if bp >= progress:
                best_t = bt
                break
        if best_t is None:
            best_t = self._best_lap_splits[-1][1]
        # Current lap time so far
        cur_t = self._recording_splits[-1][1] if self._recording_splits else 0.0
        return best_t - cur_t   # positive = we are ahead of best

    def _draw_gforce_circle(self, surface, cx, cy, radius):
        """
        Draw a live G-force friction circle.
        The dot position shows (lateral_g, long_g) as a fraction of
        the tyre's peak mu — the closer to the edge, the harder you
        are working the tyres.
        """
        # Background
        pygame.draw.circle(surface, (20, 26, 36), (cx, cy), radius)
        pygame.draw.circle(surface, (45, 55, 70), (cx, cy), radius, 1)
        # Rings at 50% and 100%
        pygame.draw.circle(surface, (35, 45, 58), (cx, cy), radius//2, 1)
        # Crosshair
        pygame.draw.line(surface, (35, 45, 58),
                         (cx - radius, cy), (cx + radius, cy), 1)
        pygame.draw.line(surface, (35, 45, 58),
                         (cx, cy - radius), (cx, cy + radius), 1)
        # Tyre limit circle (at peak mu)
        mu = (self.p.front_mu + self.p.rear_mu) * 0.5
        pygame.draw.circle(surface, (80, 40, 40), (cx, cy), radius, 1)
        # Labels
        s = small_font.render("G", True, (60, 70, 85))
        surface.blit(s, (cx + radius + 2, cy - 7))
        # Current G dot
        lat_g  = self.vehicle.lateral_accel / 9.81
        long_g = self.vehicle.long_accel    / 9.81
        scale  = radius / max(0.1, mu * 1.5)
        dot_x  = int(cx + lat_g  * scale)
        dot_y  = int(cy - long_g * scale)  # y-axis: up = acceleration
        dot_x  = max(cx - radius, min(cx + radius, dot_x))
        dot_y  = max(cy - radius, min(cy + radius, dot_y))
        # Color: green inside limit, red outside
        total_g = math.sqrt(lat_g**2 + long_g**2)
        dot_col = GREEN if total_g < mu else RED
        pygame.draw.circle(surface, dot_col, (dot_x, dot_y), 5)
        pygame.draw.circle(surface, WHITE,   (dot_x, dot_y), 5, 1)

    def _draw_track_view(self, surface):
        """Draw the scrolling track view in the centre area."""
        veh = self.vehicle

        # ---- Update camera shake trauma ----
        # Kerb hits or high-G events add trauma
        lat_g  = abs(veh.lateral_accel / 9.81)
        long_g = abs(veh.long_accel / 9.81)
        total_g = math.sqrt(lat_g**2 + long_g**2)
        if self.off_track:
            self._shake_trauma = min(1.0, self._shake_trauma + 0.25)
        elif total_g > 1.8:
            self._shake_trauma = min(1.0, self._shake_trauma + (total_g - 1.8) * 0.04)
        self._shake_trauma *= 0.88   # decay
        shake_mag = self._shake_trauma ** 2  # squared for snappier feel
        import random
        if shake_mag > 0.002:
            self._cam_shake_dx = (random.random() * 2 - 1) * shake_mag * 10
            self._cam_shake_dy = (random.random() * 2 - 1) * shake_mag * 10
        else:
            self._cam_shake_dx = 0.0
            self._cam_shake_dy = 0.0

        # ---- Update exhaust particles ----
        dt_approx = 1.0 / 60.0
        speed_ms = veh.speed
        self._exhaust_timer += dt_approx
        if speed_ms > 0.5 and self._exhaust_timer > 0.04:
            self._exhaust_timer = 0.0
            # spawn particle behind the car in world space
            ch = math.cos(veh.heading)
            sh = math.sin(veh.heading)
            ex_wx = veh.x - ch * 20
            ex_wy = veh.y - sh * 20
            spread = 1.5
            import random as _r
            pvx = -ch * 1.5 + (_r.random() - 0.5) * spread
            pvy = -sh * 1.5 + (_r.random() - 0.5) * spread
            self._exhaust_particles.append(
                [ex_wx, ex_wy, pvx, pvy, 0.55, 0.55]
            )
        # Update + cull particles
        alive = []
        for pt in self._exhaust_particles:
            pt[0] += pt[2] * dt_approx * 50
            pt[1] += pt[3] * dt_approx * 50
            pt[4] -= dt_approx
            if pt[4] > 0:
                alive.append(pt)
        self._exhaust_particles = alive[-40:]  # cap at 40 particles

        # ---- Update motion trail ----
        if speed_ms > 0.5:
            self._motion_trail.append((
                veh.x, veh.y, veh.heading, speed_ms
            ))
        if len(self._motion_trail) > 18:
            self._motion_trail.pop(0)

        # Clip rectangle for the main view
        view_rect = pygame.Rect(
            self.LEFT_PANEL_W, 0,
            WIDTH - self.LEFT_PANEL_W - self.RIGHT_PANEL_W, HEIGHT
        )
        # Apply shake to the camera blit position
        shake_x = int(self._cam_shake_dx)
        shake_y = int(self._cam_shake_dy)
        surface.set_clip(view_rect)

        # Scroll the pre-rendered track surface
        blit_x = -self.cam_x + shake_x
        blit_y = -self.cam_y + shake_y
        surface.blit(self._track_surface, (blit_x, blit_y))

        # V0.6: Speed heatmap overlay
        if self.show_heatmap:
            self.heatmap.draw(surface, self.track.centerline,
                              self.cam_x - shake_x, self.cam_y - shake_y)

        # V0.6: Ghost car
        if self.ghost.enabled and self.ghost.has_ghost:
            self.ghost.draw(surface, self.cam_x - shake_x, self.cam_y - shake_y)

        # ---- Motion blur trail ----
        n_trail = len(self._motion_trail)
        for ti, (tx_w, ty_w, th_w, spd) in enumerate(self._motion_trail):
            alpha = int(180 * (ti / max(n_trail, 1)) * min(1.0, spd / 8.0))
            if alpha < 8:
                continue
            stx = tx_w - (self.cam_x - shake_x)
            sty = ty_w - (self.cam_y - shake_y)
            ch_t = math.cos(th_w)
            sh_t = math.sin(th_w)
            sz   = 4 + (ti / max(n_trail, 1)) * 6  # grows toward present
            trail_pts = [
                (int(stx + sz * ch_t),             int(sty + sz * sh_t)),
                (int(stx - sz * 0.5 * ch_t + sz * 0.4 * sh_t),
                 int(sty - sz * 0.5 * sh_t - sz * 0.4 * ch_t)),
                (int(stx - sz * 0.5 * ch_t - sz * 0.4 * sh_t),
                 int(sty - sz * 0.5 * sh_t + sz * 0.4 * ch_t)),
            ]
            try:
                tr_surf = pygame.Surface((int(sz*4+4), int(sz*4+4)), pygame.SRCALPHA)
                tr_col  = (65, 210, 230, alpha)
                off_pts = [
                    (int(stx - (stx - sz*2) + p[0] - stx + sz*2),
                     int(sty - (sty - sz*2) + p[1] - sty + sz*2))
                    for p in trail_pts
                ]
                pygame.draw.polygon(tr_surf, tr_col, off_pts)
                surface.blit(tr_surf, (int(stx - sz*2), int(sty - sz*2)))
            except Exception:
                pass

        # Draw player vehicle on top
        self._draw_vehicle(surface)

        # Off-track overlay
        if self.off_track:
            alpha_surf = pygame.Surface(
                (view_rect.width, view_rect.height), pygame.SRCALPHA
            )
            alpha_surf.fill((240, 60, 60, 35))
            surface.blit(alpha_surf, (self.LEFT_PANEL_W, 0))

        surface.set_clip(None)

        # View border
        pygame.draw.rect(surface, (40, 48, 60), view_rect, 2)

    def cycle_camera(self):
        """Cycle between 3D Chase, 3D Cockpit FPV, and 2D Top-Down camera modes."""
        self.camera_mode = (self.camera_mode + 1) % 3

    def toggle_wide_mode(self):
        """Toggle between full-screen immersive view and split telemetry panel view."""
        self.wide_mode = not self.wide_mode

    def _init_3d_world(self):
        """Precompute 3D track boundary ribbons and trackside scenery."""
        cl = self.track.centerline
        n = self.track.n
        hw = float(self.track.TRACK_WIDTH)  # 44.0 px

        self._track_3d_geom = []
        for i in range(n):
            p_prev = cl[(i - 1) % n]
            p_curr = cl[i]
            p_next = cl[(i + 1) % n]

            # Tangent
            tx = p_next.x - p_prev.x
            ty = p_next.y - p_prev.y
            t_len = math.hypot(tx, ty)
            if t_len > 1e-6:
                tx /= t_len
                ty /= t_len
            else:
                tx, ty = 1.0, 0.0

            # Normal pointing right of travel direction
            nx = ty
            ny = -tx

            # Curvature indicator
            next_next = cl[(i + 2) % n]
            t2x = next_next.x - p_curr.x
            t2y = next_next.y - p_curr.y
            t2_len = math.hypot(t2x, t2y)
            if t2_len > 1e-6:
                t2x /= t2_len; t2y /= t2_len
                turn_rate = (tx * t2y - ty * t2x)
            else:
                turn_rate = 0.0

            # Ribbon boundary vertices in world coordinates
            cx, cy = p_curr.x, p_curr.y
            lx, ly = cx - nx * hw, cy - ny * hw
            rx, ry = cx + nx * hw, cy + ny * hw
            klx, kly = cx - nx * (hw + 11.0), cy - ny * (hw + 11.0)
            krx, kry = cx + nx * (hw + 11.0), cy + ny * (hw + 11.0)
            glx, gly = cx - nx * (hw + 320.0), cy - ny * (hw + 320.0)
            grx, gry = cx + nx * (hw + 320.0), cy + ny * (hw + 320.0)

            self._track_3d_geom.append({
                'c': (cx, cy),
                'l': (lx, ly),
                'r': (rx, ry),
                'kl': (klx, kly),
                'kr': (krx, kry),
                'gl': (glx, gly),
                'gr': (grx, gry),
                'turn': turn_rate,
                'has_kerb': abs(turn_rate) > 0.012,
                'kerb_side': -1 if turn_rate > 0 else 1
            })

        # Trackside objects: Braking boards (150, 100, 50), Apex cones, Grandstands, Trees
        self._trackside_objects = []

        apex_candidates = []
        for i in range(n):
            geom = self._track_3d_geom[i]
            prev_g = self._track_3d_geom[(i - 1) % n]
            next_g = self._track_3d_geom[(i + 1) % n]
            if abs(geom['turn']) > 0.026:
                if abs(geom['turn']) >= abs(prev_g['turn']) and abs(geom['turn']) >= abs(next_g['turn']):
                    apex_candidates.append(i)

        for apex_idx in apex_candidates:
            geom = self._track_3d_geom[apex_idx]
            # Apex cone on inside of corner
            inside_side = geom['kerb_side']
            cone_x = geom['c'][0] + (geom['r'][0] - geom['c'][0]) * (1.08 if inside_side == 1 else -1.08)
            cone_y = geom['c'][1] + (geom['r'][1] - geom['c'][1]) * (1.08 if inside_side == 1 else -1.08)
            self._trackside_objects.append({
                'type': 'cone',
                'x': cone_x, 'y': cone_y, 'z': 0.0,
                'idx': apex_idx
            })

            # Braking boards on outside of track before corner
            outside_side = -inside_side
            for dist_pts, dist_label in [(18, '150'), (12, '100'), (6, '50')]:
                entry_idx = (apex_idx - dist_pts) % n
                e_geom = self._track_3d_geom[entry_idx]
                bx = e_geom['c'][0] + (e_geom['r'][0] - e_geom['c'][0]) * (1.35 if outside_side == 1 else -1.35)
                by = e_geom['c'][1] + (e_geom['r'][1] - e_geom['c'][1]) * (1.35 if outside_side == 1 else -1.35)
                self._trackside_objects.append({
                    'type': 'board',
                    'label': dist_label,
                    'x': bx, 'y': by, 'z': 0.0,
                    'idx': entry_idx
                })

        # Grandstand along the main straight
        for s_idx in [15, 42]:
            sg = self._track_3d_geom[s_idx]
            gx = sg['c'][0] - (sg['r'][0] - sg['c'][0]) * 1.85
            gy = sg['c'][1] - (sg['r'][1] - sg['c'][1]) * 1.85
            self._trackside_objects.append({
                'type': 'grandstand',
                'x': gx, 'y': gy, 'z': 0.0,
                'idx': s_idx
            })

        # Trees & floodlight poles spaced around perimeter
        for i in range(0, n, 22):
            tg = self._track_3d_geom[i]
            side = 1 if (i // 22) % 2 == 0 else -1
            dist_mult = 1.85 + ((i * 19) % 7) * 0.16
            tx = tg['c'][0] + (tg['r'][0] - tg['c'][0]) * dist_mult * side
            ty = tg['c'][1] + (tg['r'][1] - tg['c'][1]) * dist_mult * side
            obj_type = 'pole' if (i % 66 == 0) else 'tree'
            self._trackside_objects.append({
                'type': obj_type,
                'x': tx, 'y': ty, 'z': 0.0,
                'idx': i
            })

        # Distant mountain ridge peaks (for 360-degree horizon)
        self._mountain_peaks = []
        for mi in range(48):
            ang = 2 * math.pi * mi / 48.0
            h = 42.0 + 32.0 * math.sin(mi * 0.72) + 18.0 * math.cos(mi * 1.85)
            self._mountain_peaks.append((ang, h))

    def _update_3d_physics(self, dt):
        """Update RPM, gear, flames, tire smoke, and 3D camera tracking."""
        spd_kmh = self.vehicle.speed * 3.6
        throttle = self.vehicle.throttle
        brake = self.vehicle.brake_input
        lat_g = self.vehicle.lateral_accel / 9.81
        long_g = self.vehicle.long_accel / 9.81

        # Sequential automatic transmission & RPM model
        gear_speeds = [0.0, 56.0, 94.0, 134.0, 175.0, 218.0, 265.0]
        curr_g = 1
        for g in range(1, 7):
            if spd_kmh >= gear_speeds[g-1] - 4.0:
                curr_g = g
        self._gear = curr_g

        # Realistic high-revving Formula engine RPM (4000 - 13800 RPM)
        g_min = gear_speeds[curr_g - 1]
        g_max = gear_speeds[curr_g]
        ratio = max(0.0, min(1.0, (spd_kmh - g_min) / max(1.0, g_max - g_min)))
        target_rpm = 4200.0 + ratio * 8800.0 + throttle * 800.0
        if spd_kmh < 1.0:
            target_rpm = 2500.0 + throttle * 7500.0
        self._rpm += (target_rpm - self._rpm) * min(1.0, 22.0 * dt)

        # Exhaust flame bursts on throttle lift or downshift
        if self._prev_throttle > 0.65 and throttle < 0.2 and self._rpm > 6500.0:
            self._flame_timer = 0.38
        elif self._flame_timer > 0.0:
            self._flame_timer -= dt
        self._prev_throttle = throttle

        # Tire smoke on hard braking or sliding
        if (brake > 0.75 and spd_kmh > 15.0) or (abs(lat_g) > 1.65 and spd_kmh > 25.0):
            import random
            sh = math.sin(self.vehicle.heading)
            ch = math.cos(self.vehicle.heading)
            for side in [-1, 1]:
                wx = self.vehicle.x - ch * 14 + sh * (side * 10)
                wy = self.vehicle.y - sh * 14 - ch * (side * 10)
                self._tire_smoke.append({
                    'x': wx + (random.random() - 0.5) * 4.0,
                    'y': wy + (random.random() - 0.5) * 4.0,
                    'z': random.random() * 2.0,
                    'vz': 4.0 + random.random() * 6.0,
                    'r': 3.0,
                    'alpha': 180,
                    'life': 0.45
                })

        alive_smoke = []
        for sm in self._tire_smoke:
            sm['life'] -= dt
            sm['z'] += sm['vz'] * dt
            sm['r'] += dt * 14.0
            sm['alpha'] = int(max(0, 180 * (sm['life'] / 0.45)))
            if sm['life'] > 0.0 and sm['alpha'] > 5:
                alive_smoke.append(sm)
        self._tire_smoke = alive_smoke[-45:]

        # Smooth steering wheel angle for cockpit view
        self._steer_wheel_visual += (self.vehicle.steering - self._steer_wheel_visual) * min(1.0, 18.0 * dt)

        # Camera tracking in 3D
        veh = self.vehicle
        vh = veh.heading

        if self.camera_mode == 0:
            # 3D CHASE CAMERA
            # Rock-solid, fixed follow distance and height (car never pulls away)
            chase_dist = 44.0
            chase_height = 16.5

            # Responsive heading tracking without rubber-banding or drift
            diff = (vh - self.cam_3d_heading + math.pi) % (2 * math.pi) - math.pi
            self.cam_3d_heading += diff * min(1.0, 24.0 * dt)

            target_cx = veh.x - math.cos(self.cam_3d_heading) * chase_dist
            target_cy = veh.y - math.sin(self.cam_3d_heading) * chase_dist

            self.cam_3d_x = target_cx
            self.cam_3d_y = target_cy
            self.cam_3d_z = chase_height + (self._shake_trauma ** 2) * 3.0
            self.cam_3d_pitch = -long_g * 0.0015
            self.cam_3d_roll  = lat_g * 0.025

        elif self.camera_mode == 1:
            # 3D COCKPIT FPV CAMERA
            ch = math.cos(vh)
            sh = math.sin(vh)
            self.cam_3d_x = veh.x + ch * 4.0
            self.cam_3d_y = veh.y + sh * 4.0
            self.cam_3d_z = 10.5 + (self._shake_trauma ** 2) * 4.0

            apex_lean = veh.steering * 0.20
            self.cam_3d_heading = vh + apex_lean
            self.cam_3d_pitch = -long_g * 0.002
            self.cam_3d_roll  = lat_g * 0.035

    def _draw_3d_view(self, surface, view_rect):
        """Render the full 3D pseudo-3D perspective road and cockpit/chase view."""
        surface.set_clip(view_rect)

        vx, vy, vw, vh = view_rect.x, view_rect.y, view_rect.width, view_rect.height
        vp_cx = vx + vw // 2
        vp_cy = vy + int(vh * 0.44) + int(self.cam_3d_pitch * 150.0)
        fov = 580.0
        fwd_near = 18.0 if self.camera_mode == 0 else 6.0

        cos_h = math.cos(self.cam_3d_heading)
        sin_h = math.sin(self.cam_3d_heading)
        cam_x = self.cam_3d_x
        cam_y = self.cam_3d_y
        cam_z = self.cam_3d_z

        def world_to_cam(wx, wy, wz):
            dx = wx - cam_x
            dy = wy - cam_y
            dz = wz - cam_z
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            up  = dz
            return fwd, lat, up

        def cam_to_screen(fwd, lat, up):
            inv_f = fov / max(fwd, 0.5)
            sx = vp_cx + lat * inv_f
            sy = vp_cy - up * inv_f
            return sx, sy

        def clip_edge(p0, p1):
            f0, l0, u0 = world_to_cam(*p0)
            f1, l1, u1 = world_to_cam(*p1)
            if f0 < fwd_near and f1 < fwd_near:
                return None, None
            if f0 < fwd_near:
                t = (fwd_near - f0) / max(0.001, (f1 - f0))
                f0 = fwd_near
                l0 = l0 + t * (l1 - l0)
                u0 = u0 + t * (u1 - u0)
            elif f1 < fwd_near:
                t = (fwd_near - f1) / max(0.001, (f0 - f1))
                f1 = fwd_near
                l1 = l1 + t * (l0 - l1)
                u1 = u1 + t * (u0 - u1)
            sx0, sy0 = cam_to_screen(f0, l0, u0)
            sx1, sy1 = cam_to_screen(f1, l1, u1)
            sx0 = max(-600, min(vw + 600, sx0))
            sx1 = max(-600, min(vw + 600, sx1))
            sy0 = max(-100, min(vh + 300, sy0))
            sy1 = max(-100, min(vh + 300, sy1))
            return (sx0, sy0), (sx1, sy1)

        def project_3d(wx, wy, wz):
            fwd, lat, up = world_to_cam(wx, wy, wz)
            if fwd < fwd_near * 0.8:
                return None
            sx, sy = cam_to_screen(fwd, lat, up)
            return (sx, sy, fwd)

        # ----------------------------------------------------
        # 1. SKY GRADIENT & PARALLAX MOUNTAINS
        # ----------------------------------------------------
        n_sky_bands = 12
        for bi in range(n_sky_bands):
            y0 = vy + int((vp_cy - vy) * bi / float(n_sky_bands))
            y1 = vy + int((vp_cy - vy) * (bi + 1) / float(n_sky_bands))
            ratio = bi / float(n_sky_bands)
            r = int(12 + ratio * 22)
            g = int(16 + ratio * 30)
            b = int(28 + ratio * 44)
            pygame.draw.rect(surface, (r, g, b), (vx, y0, vw, max(1, y1 - y0)))

        # Distant mountain silhouettes
        m_pts = [(vx, vp_cy)]
        for ang, mh in self._mountain_peaks:
            rel = (ang - self.cam_3d_heading + math.pi) % (2 * math.pi) - math.pi
            if -1.4 < rel < 1.4:
                px = vp_cx + rel * (vw * 1.15 / math.pi)
                py = vp_cy - int(mh * 1.1)
                m_pts.append((px, py))
        m_pts.append((vx + vw, vp_cy))
        if len(m_pts) >= 3:
            pygame.draw.polygon(surface, (22, 30, 44), m_pts)

        # Horizon treeline band
        pygame.draw.rect(surface, (18, 25, 34), (vx, vp_cy - 8, vw, 8))

        # Ground terrain base below horizon
        pygame.draw.rect(surface, (18, 36, 22), (vx, vp_cy, vw, vy + vh - vp_cy))

        # ----------------------------------------------------
        # 2. 3D TRACK SEGMENTS (Continuous Horizon-to-Bottom Projection)
        # ----------------------------------------------------
        n = self.track.n
        s_min = -8 if self.camera_mode == 0 else 0
        for s in range(160, s_min - 1, -1):
            i0 = (self.closest_idx + s) % n
            i1 = (self.closest_idx + s + 1) % n
            g0 = self._track_3d_geom[i0]
            g1 = self._track_3d_geom[i1]

            # Clip and project road boundaries with near-plane interpolation
            l0, l1 = clip_edge((g0['l'][0], g0['l'][1], 0.0), (g1['l'][0], g1['l'][1], 0.0))
            r0, r1 = clip_edge((g0['r'][0], g0['r'][1], 0.0), (g1['r'][0], g1['r'][1], 0.0))

            if l0 and l1 and r0 and r1:
                c0, c1 = clip_edge((g0['c'][0], g0['c'][1], 0.0), (g1['c'][0], g1['c'][1], 0.0))
                kl0, kl1 = clip_edge((g0['kl'][0], g0['kl'][1], 0.0), (g1['kl'][0], g1['kl'][1], 0.0))
                kr0, kr1 = clip_edge((g0['kr'][0], g0['kr'][1], 0.0), (g1['kr'][0], g1['kr'][1], 0.0))
                gl0, gl1 = clip_edge((g0['gl'][0], g0['gl'][1], -2.0), (g1['gl'][0], g1['gl'][1], -2.0))
                gr0, gr1 = clip_edge((g0['gr'][0], g0['gr'][1], -2.0), (g1['gr'][0], g1['gr'][1], -2.0))

                approx_dist = max(1.0, s * 3.8 + 44.0)

                # Grass Runoff Left & Right (crisp alternating turf bands)
                if gl0 and gl1:
                    g_col = (26, 74, 36) if (i0 // 4) % 2 == 0 else (20, 58, 28)
                    pygame.draw.polygon(surface, g_col, [gl1, l1, l0, gl0])
                if gr0 and gr1:
                    g_col = (26, 74, 36) if (i0 // 4) % 2 == 0 else (20, 58, 28)
                    pygame.draw.polygon(surface, g_col, [r1, gr1, gr0, r0])

                # 3D Rumble Kerbs (High-Contrast FIA Red & White)
                if g0['has_kerb'] or g1['has_kerb'] or (s < 90 and abs(g0['turn']) > 0.008):
                    kerb_col = (235, 35, 35) if (i0 // 3) % 2 == 0 else (255, 255, 255)
                    if kl0 and kl1:
                        pygame.draw.polygon(surface, kerb_col, [kl1, l1, l0, kl0])
                    if kr0 and kr1:
                        pygame.draw.polygon(surface, kerb_col, [r1, kr1, kr0, r0])

                # Asphalt Road Surface (Vibrant High-Contrast Racing Slate)
                base_val = 38 if (i0 // 4) % 2 == 0 else 46
                ar, ag, ab = base_val, base_val + 2, base_val + 6

                # Speed Heatmap tinting
                if self.show_heatmap:
                    h_spd = self.heatmap.best[i0] if self.heatmap.best[i0] > 0 else self.heatmap._current[i0]
                    if h_spd > 0:
                        spd_rat = min(1.0, h_spd / max(1.0, self.max_speed_seen or 200.0))
                        ar = int(ar * 0.60 + (240 if spd_rat < 0.4 else (225 if spd_rat < 0.7 else 45)) * 0.40)
                        ag = int(ag * 0.60 + (55  if spd_rat < 0.4 else (210 if spd_rat < 0.7 else 225)) * 0.40)
                        ab = int(ab * 0.60 + (55  if spd_rat < 0.4 else (55  if spd_rat < 0.7 else 125)) * 0.40)

                pygame.draw.polygon(surface, (ar, ag, ab), [l1, r1, r0, l0])

                # Dark rubbered-in racing groove along centerline
                if c0 and c1:
                    pygame.draw.line(surface, (max(16, ar - 12), max(18, ag - 12), max(22, ab - 12)), c1, c0, max(2, int(6.0 * (fov / max(approx_dist, 10.0)))))

                # Start / Finish line checkered pattern
                if i0 == 0 or i1 == 0:
                    for chk in range(8):
                        t_a = chk / 8.0
                        t_b = (chk + 1) / 8.0
                        p_a0 = (l0[0] + (r0[0] - l0[0]) * t_a, l0[1] + (r0[1] - l0[1]) * t_a)
                        p_b0 = (l0[0] + (r0[0] - l0[0]) * t_b, l0[1] + (r0[1] - l0[1]) * t_b)
                        p_a1 = (l1[0] + (r1[0] - l1[0]) * t_a, l1[1] + (r1[1] - l1[1]) * t_a)
                        p_b1 = (l1[0] + (r1[0] - l1[0]) * t_b, l1[1] + (r1[1] - l1[1]) * t_b)
                        chk_col = (255, 255, 255) if chk % 2 == 0 else (18, 20, 26)
                        pygame.draw.polygon(surface, chk_col, [p_a1, p_b1, p_b0, p_a0])

                # High-contrast 4px solid white boundary edge lines
                pygame.draw.line(surface, (255, 255, 255), l1, l0, 4)
                pygame.draw.line(surface, (255, 255, 255), r1, r0, 4)

                # Centerline dashes (Bright Racing Yellow)
                if (i0 // 4) % 2 == 0 and c0 and c1:
                    pygame.draw.line(surface, (255, 240, 95), c1, c0, 2)

        # ----------------------------------------------------
        # 3. DEPTH-SORTED 3D OBJECTS & VEHICLES
        # ----------------------------------------------------
        render_list = []

        # Trackside scenery objects
        for obj in self._trackside_objects:
            res = project_3d(obj['x'], obj['y'], obj['z'])
            if res and 3.5 < res[2] < 380.0:
                render_list.append((res[2], obj['type'], obj, res))

        # Ghost Car in 3D
        if self.ghost.enabled and self.ghost.has_ghost and self.ghost._ghost:
            g_res = project_3d(self.ghost.ghost_x, self.ghost.ghost_y, 0.0)
            if g_res and g_res[2] > 2.0:
                render_list.append((g_res[2], 'ghost_car', None, g_res))

        # Player Car in 3D (Chase View)
        if self.camera_mode == 0:
            p_res = project_3d(self.vehicle.x, self.vehicle.y, 0.0)
            if p_res and p_res[2] > 2.0:
                render_list.append((p_res[2], 'player_car', None, p_res))

        # Tire smoke particles
        for sm in self._tire_smoke:
            s_res = project_3d(sm['x'], sm['y'], sm['z'])
            if s_res and s_res[2] > 2.0:
                render_list.append((s_res[2], 'smoke', sm, s_res))

        # Sort render list by depth descending (furthest first)
        render_list.sort(key=lambda item: item[0], reverse=True)

        for depth, otype, odata, (sx, sy, fwd) in render_list:
            scale = fov / fwd

            if otype == 'tree':
                tw = max(2, int(4 * scale))
                th = max(6, int(30 * scale))
                pygame.draw.rect(surface, (60, 42, 28), (sx - tw//2, sy - th, tw, th))
                for ti in range(3):
                    tier_w = max(4, int((17 - ti * 4) * scale))
                    tier_h = max(3, int(11 * scale))
                    tier_y = sy - th + int(ti * 7 * scale)
                    pts = [(sx, tier_y - tier_h), (sx - tier_w, tier_y), (sx + tier_w, tier_y)]
                    pygame.draw.polygon(surface, (16 + ti * 5, 50 + ti * 8, 24 + ti * 5), pts)

            elif otype == 'pole':
                pw = max(2, int(2.5 * scale))
                ph = max(10, int(38 * scale))
                pygame.draw.rect(surface, (120, 130, 145), (sx - pw//2, sy - ph, pw, ph))
                # Floodlight head
                lw = max(4, int(9 * scale))
                lh = max(2, int(4 * scale))
                pygame.draw.rect(surface, (230, 235, 245), (sx - lw//2, sy - ph - lh, lw, lh))

            elif otype == 'grandstand':
                gw = max(18, int(52 * scale))
                gh = max(12, int(34 * scale))
                # Base tiered stands
                pygame.draw.rect(surface, (65, 75, 90), (sx - gw//2, sy - gh, gw, gh))
                # Red/white striped canopy roof
                rw = int(gw * 1.15)
                rh = max(4, int(8 * scale))
                roof_rect = pygame.Rect(sx - rw//2, sy - gh - rh, rw, rh)
                pygame.draw.rect(surface, (220, 45, 45), roof_rect)
                pygame.draw.rect(surface, (240, 240, 245), roof_rect, 1)

            elif otype == 'board':
                bw = max(8, int(22 * scale))
                bh = max(10, int(26 * scale))
                post_w = max(2, int(3 * scale))
                post_h = max(8, int(18 * scale))
                pygame.draw.rect(surface, (80, 85, 95), (sx - post_w//2, sy - post_h, post_w, post_h))
                board_rect = pygame.Rect(sx - bw//2, sy - post_h - bh, bw, bh)
                pygame.draw.rect(surface, (240, 242, 245), board_rect, border_radius=max(1, int(2*scale)))
                pygame.draw.rect(surface, (20, 25, 35), board_rect, max(1, int(1.5*scale)), border_radius=max(1, int(2*scale)))
                if scale > 0.8:
                    lbl_s = small_font.render(odata['label'], True, (15, 20, 30))
                    surface.blit(lbl_s, (sx - lbl_s.get_width()//2, sy - post_h - bh//2 - lbl_s.get_height()//2))

            elif otype == 'cone':
                cw = max(3, int(8 * scale))
                ch = max(5, int(15 * scale))
                pts = [(sx, sy - ch), (sx - cw, sy), (sx + cw, sy)]
                pygame.draw.polygon(surface, (255, 110, 20), pts)
                # Reflective white band
                b_y = sy - int(ch * 0.45)
                pygame.draw.line(surface, WHITE, (sx - int(cw*0.55), b_y), (sx + int(cw*0.55), b_y), max(1, int(1.8*scale)))

            elif otype == 'smoke':
                sm = odata
                sm_r = max(2, int(sm['r'] * scale))
                try:
                    sm_surf = pygame.Surface((sm_r * 2, sm_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(sm_surf, (220, 225, 235, sm['alpha']), (sm_r, sm_r), sm_r)
                    surface.blit(sm_surf, (sx - sm_r, sy - sm_r))
                except Exception:
                    pass

            elif otype == 'ghost_car':
                # Neon Cyan Holographic Formula Car
                gw = int(22 * scale)
                gh = int(9 * scale)
                ghost_pts = [
                    (sx - int(10*scale), sy - int(3*scale)),
                    (sx + int(10*scale), sy - int(3*scale)),
                    (sx + int(6*scale), sy - int(9*scale)),
                    (sx - int(6*scale), sy - int(9*scale)),
                ]
                try:
                    gh_surf = pygame.Surface((gw * 2 + 10, gh * 2 + 10), pygame.SRCALPHA)
                    pygame.draw.polygon(gh_surf, (65, 210, 230, 110), [(p[0] - sx + gw, p[1] - sy + gh) for p in ghost_pts])
                    surface.blit(gh_surf, (sx - gw, sy - gh))
                except Exception:
                    pass
                pygame.draw.polygon(surface, (120, 235, 255), ghost_pts, 2)
                # Ghost rear wing
                pygame.draw.line(surface, CYAN, (sx - int(11*scale), sy - int(10*scale)), (sx + int(11*scale), sy - int(10*scale)), 2)

            elif otype == 'player_car':
                # 3D Formula Car Model from behind (Chase View)
                roll_px = int(self.cam_3d_roll * 18.0 * scale)

                # Rear Tyres
                for side in [-1, 1]:
                    tx = sx + side * int(11.5 * scale)
                    ty = sy - int(2.5 * scale)
                    tw = max(3, int(4.6 * scale))
                    th = max(6, int(9.5 * scale))
                    pygame.draw.rect(surface, (16, 18, 22), (tx - tw//2, ty - th//2, tw, th), border_radius=max(1, int(1.5*scale)))
                    pygame.draw.circle(surface, (110, 115, 130), (tx, ty), max(1, int(1.8*scale)))
                    # Tread rotation highlights
                    t_off = int((pygame.time.get_ticks() * (self.vehicle.speed / 10.0)) % 8)
                    pygame.draw.line(surface, (45, 50, 60), (tx - tw//2 + 1, ty - th//2 + t_off), (tx + tw//2 - 1, ty - th//2 + t_off), 1)

                # Double Wishbone Suspension Arms
                for side in [-1, 1]:
                    tx = sx + side * int(11.5 * scale)
                    ty = sy - int(2.5 * scale)
                    pygame.draw.line(surface, (90, 95, 110), (sx + side * int(4*scale), sy - int(3*scale) + roll_px), (tx, ty - int(2*scale)), 1)
                    pygame.draw.line(surface, (90, 95, 110), (sx + side * int(4*scale), sy - int(1*scale) + roll_px), (tx, ty + int(2*scale)), 1)

                # Diffuser with vertical fences
                dw = int(14 * scale)
                dh = int(3.5 * scale)
                pygame.draw.rect(surface, (22, 26, 34), (sx - dw//2, sy - dh, dw, dh))
                for fi in [-3, -1, 1, 3]:
                    fx = sx + int(fi * 1.6 * scale)
                    pygame.draw.line(surface, (14, 16, 20), (fx, sy - dh), (fx, sy), 1)

                # FIA Rain / Brake Light
                is_braking = self.vehicle.brake_input > 0.3
                flash_on = (pygame.time.get_ticks() % 240 < 120) or is_braking
                rl_col = (255, 30, 30) if flash_on else (120, 20, 20)
                rl_w = max(2, int(2.4 * scale))
                rl_h = max(2, int(2.2 * scale))
                pygame.draw.rect(surface, rl_col, (sx - rl_w//2, sy - int(2.4*scale) + roll_px, rl_w, rl_h))
                if is_braking or flash_on:
                    pygame.draw.circle(surface, (255, 60, 60), (sx, sy - int(1.3*scale) + roll_px), max(2, int(3.5*scale)), 1)

                # Exhaust tailpipes & flame backfires
                for ex_side in [-1, 1]:
                    ex_x = sx + ex_side * int(2.8 * scale)
                    ex_y = sy - int(3.2 * scale) + roll_px
                    pygame.draw.circle(surface, (60, 65, 75), (ex_x, ex_y), max(1, int(1.4*scale)))
                    if self._flame_timer > 0.0:
                        # Spitting flame burst
                        fl_pts = [
                            (ex_x - int(1.5*scale), ex_y),
                            (ex_x + int(1.5*scale), ex_y),
                            (ex_x, ex_y + int(4.5*scale))
                        ]
                        pygame.draw.polygon(surface, (255, 140, 30), fl_pts)
                        pygame.draw.polygon(surface, (120, 200, 255), [(ex_x - int(0.8*scale), ex_y), (ex_x + int(0.8*scale), ex_y), (ex_x, ex_y + int(2.2*scale))])

                # Main Monocoque Body
                body_pts = [
                    (sx - int(7.5*scale), sy - int(3*scale) + roll_px),
                    (sx + int(7.5*scale), sy - int(3*scale) + roll_px),
                    (sx + int(5.5*scale), sy - int(9.5*scale) + roll_px),
                    (sx - int(5.5*scale), sy - int(9.5*scale) + roll_px),
                ]
                pygame.draw.polygon(surface, (42, 52, 70), body_pts)
                pygame.draw.polygon(surface, (65, 80, 105), body_pts, 1)

                # Livery stripe & Shark Fin
                pygame.draw.line(surface, (50, 140, 230), (sx, sy - int(3*scale) + roll_px), (sx, sy - int(10*scale) + roll_px), max(1, int(1.8*scale)))

                # Driver Helmet with dynamic corner lean
                helm_lean = int(self.vehicle.lateral_accel / 9.81 * 2.5 * scale)
                helm_x = sx + helm_lean
                helm_y = sy - int(8.0 * scale) + roll_px
                helm_r = max(2, int(2.6 * scale))
                pygame.draw.circle(surface, (235, 75, 35), (helm_x, helm_y), helm_r)
                pygame.draw.circle(surface, (255, 140, 70), (helm_x, helm_y), helm_r, 1)

                # Halo Safety Arch
                halo_y = sy - int(9.2 * scale) + roll_px
                pygame.draw.line(surface, (150, 160, 175), (sx - int(4*scale), halo_y + int(2*scale)), (sx, halo_y), max(1, int(1.8*scale)))
                pygame.draw.line(surface, (150, 160, 175), (sx + int(4*scale), halo_y + int(2*scale)), (sx, halo_y), max(1, int(1.8*scale)))

                # Multi-element Rear Wing with DRS
                drs_active = getattr(getattr(self, '_nuclear_aero', None), 'drs_active', False)
                rw_w = int(22 * scale)
                rw_y = sy - int(11.2 * scale) + roll_px
                # Main lower plane
                pygame.draw.line(surface, (28, 35, 48), (sx - rw_w//2, rw_y), (sx + rw_w//2, rw_y), max(2, int(2.6*scale)))
                # Upper DRS Flap
                flap_y = rw_y - max(2, int(2.5 * scale))
                flap_col = (30, 220, 80) if drs_active else (220, 45, 45)
                pygame.draw.line(surface, flap_col, (sx - rw_w//2 + 2, flap_y), (sx + rw_w//2 - 2, flap_y), max(1, int(2.0*scale)))
                # Endplates
                for ep_side in [-1, 1]:
                    ep_x = sx + ep_side * rw_w//2
                    pygame.draw.line(surface, (50, 140, 230), (ep_x, rw_y - int(3*scale)), (ep_x, rw_y + int(3.5*scale)), max(1, int(2.2*scale)))

        # ----------------------------------------------------
        # 4. COCKPIT / IN-CAR FIRST-PERSON VIEW OVERLAY (FPV)
        # ----------------------------------------------------
        if self.camera_mode == 1:
            cw_cx = vp_cx
            cw_cy = vy + vh + 10

            # Front Open Wheels visible on left and right
            spd_k = self.vehicle.speed * 3.6
            st_deg = math.degrees(self.vehicle.steering)
            t_rot = int((pygame.time.get_ticks() * (self.vehicle.speed / 8.0)) % 10)

            # Left front wheel
            l_wheel_x = vx + int(vw * 0.16) + int(st_deg * 0.4)
            l_wheel_y = vy + int(vh * 0.72)
            pygame.draw.rect(surface, (16, 18, 22), (l_wheel_x - 18, l_wheel_y - 48, 36, 96), border_radius=6)
            pygame.draw.circle(surface, (110, 115, 130), (l_wheel_x, l_wheel_y), 15)
            pygame.draw.circle(surface, (160, 165, 180), (l_wheel_x, l_wheel_y), 8)
            # Wishbones
            pygame.draw.line(surface, (60, 65, 80), (l_wheel_x + 16, l_wheel_y - 20), (vx + int(vw * 0.32), vy + int(vh * 0.80)), 3)
            pygame.draw.line(surface, (60, 65, 80), (l_wheel_x + 16, l_wheel_y + 20), (vx + int(vw * 0.32), vy + int(vh * 0.86)), 3)

            # Right front wheel
            r_wheel_x = vx + int(vw * 0.84) + int(st_deg * 0.4)
            r_wheel_y = vy + int(vh * 0.72)
            pygame.draw.rect(surface, (16, 18, 22), (r_wheel_x - 18, r_wheel_y - 48, 36, 96), border_radius=6)
            pygame.draw.circle(surface, (110, 115, 130), (r_wheel_x, r_wheel_y), 15)
            pygame.draw.circle(surface, (160, 165, 180), (r_wheel_x, r_wheel_y), 8)
            # Wishbones
            pygame.draw.line(surface, (60, 65, 80), (r_wheel_x - 16, r_wheel_y - 20), (vx + int(vw * 0.68), vy + int(vh * 0.80)), 3)
            pygame.draw.line(surface, (60, 65, 80), (r_wheel_x - 16, r_wheel_y + 20), (vx + int(vw * 0.68), vy + int(vh * 0.86)), 3)

            # Cockpit chassis tub surround (lower rim)
            tub_pts = [
                (vx, vy + vh),
                (vx, vy + int(vh * 0.86)),
                (vx + int(vw * 0.28), vy + int(vh * 0.78)),
                (vx + int(vw * 0.50), vy + int(vh * 0.72)),
                (vx + int(vw * 0.72), vy + int(vh * 0.78)),
                (vx + vw, vy + int(vh * 0.86)),
                (vx + vw, vy + vh),
            ]
            pygame.draw.polygon(surface, (20, 24, 32), tub_pts)
            pygame.draw.polygon(surface, (45, 55, 72), tub_pts, 2)

            # Formula Butterfly Steering Wheel
            sw_rot = self._steer_wheel_visual * 1.4
            sw_cx = vp_cx + int(sw_rot * 18.0)
            sw_cy = vy + vh - 68

            # Wheel grips (left and right contoured Alcantara handles)
            cos_sw = math.cos(sw_rot)
            sin_sw = math.sin(sw_rot)
            for side in [-1, 1]:
                gx_raw = side * 105
                gy_raw = -10
                gx = sw_cx + gx_raw * cos_sw - gy_raw * sin_sw
                gy = sw_cy + gx_raw * sin_sw + gy_raw * cos_sw
                pygame.draw.ellipse(surface, (14, 16, 20), (gx - 16, gy - 42, 32, 84))
                pygame.draw.ellipse(surface, (42, 48, 60), (gx - 16, gy - 42, 32, 84), 2)

            # Wheel carbon center plate
            cplate_rect = pygame.Rect(sw_cx - 85, sw_cy - 60, 170, 95)
            pygame.draw.rect(surface, (24, 28, 36), cplate_rect, border_radius=8)
            pygame.draw.rect(surface, (55, 65, 84), cplate_rect, 2, border_radius=8)

            # 10-LED Shift Light Array
            for i in range(10):
                lx = sw_cx - 68 + i * 15
                ly = sw_cy - 72
                led_rpm_thresh = 4000.0 + i * 920.0
                led_on = self._rpm >= led_rpm_thresh
                is_redline = self._rpm > 12800.0
                if is_redline and (pygame.time.get_ticks() % 160 < 80):
                    col = (80, 180, 255)
                elif led_on:
                    col = (60, 225, 100) if i < 4 else ((245, 50, 50) if i < 8 else (70, 160, 255))
                else:
                    col = (35, 42, 54)
                pygame.draw.circle(surface, col, (lx, ly), 5)
                pygame.draw.circle(surface, (15, 18, 24), (lx, ly), 5, 1)

            # Digital OLED Display Screen on Wheel
            disp_rect = pygame.Rect(sw_cx - 65, sw_cy - 52, 130, 72)
            pygame.draw.rect(surface, (8, 12, 18), disp_rect, border_radius=4)
            pygame.draw.rect(surface, (45, 55, 75), disp_rect, 1, border_radius=4)

            # Gear display (huge in center)
            g_txt = big_font.render(str(self._gear), True, (245, 245, 250))
            surface.blit(g_txt, (sw_cx - g_txt.get_width()//2, sw_cy - 48))

            # Speedometer (km/h)
            spd_txt = font.render(f"{spd_k:03.0f} KM/H", True, CYAN)
            surface.blit(spd_txt, (sw_cx - spd_txt.get_width()//2, sw_cy - 12))

            # Delta / Lap status
            d_col = GREEN if (self.lateral_error < 10) else RED
            d_txt = small_font.render(f"RPM {int(self._rpm)}", True, YELLOW)
            surface.blit(d_txt, (sw_cx - d_txt.get_width()//2, sw_cy + 6))

            # Throttle & Brake vertical LED bars on sides of display
            tb_h = 44
            th_fill = int(self.vehicle.throttle * tb_h)
            br_fill = int(self.vehicle.brake_input * tb_h)
            # Throttle (Green, right side)
            pygame.draw.rect(surface, (20, 30, 24), (disp_rect.right - 10, disp_rect.y + 12, 6, tb_h))
            if th_fill > 0:
                pygame.draw.rect(surface, GREEN, (disp_rect.right - 10, disp_rect.y + 12 + tb_h - th_fill, 6, th_fill))
            # Brake (Red, left side)
            pygame.draw.rect(surface, (30, 20, 20), (disp_rect.left + 4, disp_rect.y + 12, 6, tb_h))
            if br_fill > 0:
                pygame.draw.rect(surface, RED, (disp_rect.left + 4, disp_rect.y + 12 + tb_h - br_fill, 6, br_fill))

            # High-speed wind vignette streaks (> 120 km/h)
            if spd_k > 120.0:
                import random
                n_streaks = int((spd_k - 120.0) / 15.0)
                for _ in range(n_streaks):
                    sx1 = vx + random.randint(0, vw)
                    sy1 = vy + random.randint(0, int(vh * 0.6))
                    slen = random.randint(15, 45)
                    pygame.draw.line(surface, (200, 220, 245, 80), (sx1, sy1), (sx1 + (random.random()-0.5)*10, sy1 + slen), 1)

        # ----------------------------------------------------
        # 5. CAMERA BADGE & HUD NOTIFICATIONS
        # ----------------------------------------------------
        mode_label = "3D CHASE CAM" if self.camera_mode == 0 else "3D COCKPIT (FPV)"
        badge_surf = small_font.render(f"[C] CAMERA: {mode_label}  |  [TAB] WIDE VIEW", True, (160, 210, 255))
        bx = vp_cx - badge_surf.get_width()//2
        by = vy + 10
        pygame.draw.rect(surface, (14, 20, 30), (bx - 8, by - 3, badge_surf.get_width() + 16, badge_surf.get_height() + 6), border_radius=4)
        pygame.draw.rect(surface, (45, 65, 95), (bx - 8, by - 3, badge_surf.get_width() + 16, badge_surf.get_height() + 6), 1, border_radius=4)
        surface.blit(badge_surf, (bx, by))

        # Off-track red border warning
        if self.off_track:
            alpha_surf = pygame.Surface((vw, vh), pygame.SRCALPHA)
            alpha_surf.fill((240, 50, 50, 40))
            surface.blit(alpha_surf, (vx, vy))

        surface.set_clip(None)
        pygame.draw.rect(surface, (40, 48, 60), view_rect, 2)

    def _draw_wide_hud(self, surface):
        """Floating sleek F1 telemetry HUD when in full-screen Wide Mode."""
        lt = self.lap_timer
        spd_k = self.vehicle.speed * 3.6

        # Top-left Floating Lap Timer Card
        card_w, card_h = 240, 88
        pygame.draw.rect(surface, (14, 18, 26), (16, 16, card_w, card_h), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (16, 16, card_w, card_h), 1, border_radius=6)
        # Lap & Time
        cur_t = LapTimer.format_time(lt.current_lap_time)
        surface.blit(small_font.render(f"LAP {lt.lap_number}  |  S{lt.current_sector}", True, GREY), (26, 22))
        surface.blit(big_font.render(cur_t, True, CYAN), (26, 38))
        best_t = LapTimer.format_time(lt.best_lap_time)
        surface.blit(small_font.render(f"BEST {best_t}", True, YELLOW), (26, 72))

        # Bottom-right Floating Telemetry & Speed Card
        sp_w, sp_h = 220, 96
        sx = WIDTH - sp_w - 16
        sy = HEIGHT - sp_h - 16
        pygame.draw.rect(surface, (14, 18, 26), (sx, sy, sp_w, sp_h), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (sx, sy, sp_w, sp_h), 1, border_radius=6)
        # Big Speed
        surface.blit(big_font.render(f"{spd_k:05.1f}", True, CYAN), (sx + 14, sy + 10))
        surface.blit(small_font.render("KM/H", True, GREY), (sx + 140, sy + 22))
        # Gear
        surface.blit(font.render(f"GEAR {self._gear}", True, WHITE), (sx + 14, sy + 48))
        # Bars
        self._gauge_bar(surface, sx + 14, sy + 74, 85, 8, self.vehicle.throttle, GREEN)
        self._gauge_bar(surface, sx + 115, sy + 74, 85, 8, self.vehicle.brake_input, RED)

        # Bottom-left Compact Minimap
        map_size = 140
        map_rect = pygame.Rect(16, HEIGHT - map_size - 16, map_size, map_size)
        self._draw_minimap(surface, map_rect)

        # Top Center Mode Notification
        mode_label = "3D CHASE CAM" if self.camera_mode == 0 else ("3D COCKPIT (FPV)" if self.camera_mode == 1 else "2D TOP-DOWN")
        t_note = small_font.render(f"WIDE VIEW  |  C=Camera [{mode_label}]  |  TAB=Exit Wide", True, (160, 200, 240))
        tx = WIDTH // 2 - t_note.get_width() // 2
        pygame.draw.rect(surface, (14, 18, 28), (tx - 10, 10, t_note.get_width() + 20, 24), border_radius=4)
        pygame.draw.rect(surface, (45, 60, 85), (tx - 10, 10, t_note.get_width() + 20, 24), 1, border_radius=4)
        surface.blit(t_note, (tx, 14))

    def _draw_minimap(self, surface, rect):
        """Draw the minimap inside the given Rect."""
        pygame.draw.rect(surface, (14, 18, 24), rect, border_radius=8)
        pygame.draw.rect(surface, (48, 56, 68), rect, 1, border_radius=8)

        # Compute bounding box of track
        xs = [p.x for p in self.track.centerline]
        ys = [p.y for p in self.track.centerline]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        tw = max(1.0, max_x - min_x)
        th = max(1.0, max_y - min_y)

        pad = 12
        map_w = rect.width  - pad * 2
        map_h = rect.height - pad * 2
        map_scale = min(map_w / tw, map_h / th)
        off_x = rect.x + pad + (map_w - tw * map_scale) / 2
        off_y = rect.y + pad + (map_h - th * map_scale) / 2

        def world_to_map(wx, wy):
            return (
                int(off_x + (wx - min_x) * map_scale),
                int(off_y + (wy - min_y) * map_scale)
            )

        # Track centerline
        map_pts = [world_to_map(p.x, p.y)
                   for p in self.track.centerline]
        if len(map_pts) >= 2:
            pygame.draw.lines(surface, (60, 70, 90), True, map_pts, 3)

        # Sector markers
        sec_cols = [YELLOW, ORANGE, PURPLE]
        for si, idx in enumerate(self.track.sector_indices):
            pt = self.track.centerline[idx]
            mx, my = world_to_map(pt.x, pt.y)
            pygame.draw.circle(surface, sec_cols[si], (mx, my), 5)

        # Start/finish
        sf = world_to_map(
            self.track.centerline[0].x,
            self.track.centerline[0].y
        )
        pygame.draw.circle(surface, GREEN, sf, 5)

        # Vehicle position
        vx_m, vy_m = world_to_map(self.vehicle.x, self.vehicle.y)
        pygame.draw.circle(surface, CYAN, (vx_m, vy_m), 5)
        # Heading tick
        hlen = 8
        hx = int(vx_m + math.cos(self.vehicle.heading) * hlen)
        hy = int(vy_m + math.sin(self.vehicle.heading) * hlen)
        pygame.draw.line(surface, WHITE, (vx_m, vy_m), (hx, hy), 2)

    def _gauge_bar(self, surface, x, y, w, h, value, color):
        """Draw a horizontal filled gauge bar."""
        pygame.draw.rect(surface, (28, 35, 45),
                         (x, y, w, h), border_radius=3)
        filled = int(max(0.0, min(1.0, value)) * w)
        if filled > 0:
            pygame.draw.rect(surface, color,
                             (x, y, filled, h), border_radius=3)

    def _draw_left_panel(self, surface):
        """Live telemetry panel on the left."""
        W = self.LEFT_PANEL_W
        if self.camera_mode == 2:
            rect = pygame.Rect(0, 0, W, HEIGHT)
            pygame.draw.rect(surface, (16, 20, 28), rect)
            pygame.draw.rect(surface, (40, 50, 65), rect, 1)
        else:
            panel_surf = pygame.Surface((W, HEIGHT), pygame.SRCALPHA)
            panel_surf.fill((12, 16, 24, 215))
            surface.blit(panel_surf, (0, 0))
            pygame.draw.line(surface, (45, 60, 85), (W, 0), (W, HEIGHT), 1)

        # Title
        surf = medium_font.render("TELEMETRY", True, CYAN)
        surface.blit(surf, (W//2 - surf.get_width()//2, 12))
        pygame.draw.line(surface, (40, 55, 70), (8, 40), (W - 8, 40), 1)

        speed_kmh = self.vehicle.speed * 3.6

        # Big speed display
        spd_surf = big_font.render(
            f"{speed_kmh:05.1f}", True, CYAN
        )
        surface.blit(spd_surf, (W//2 - spd_surf.get_width()//2, 50))
        unit_surf = small_font.render("km/h", True, GREY)
        surface.blit(unit_surf, (W//2 - unit_surf.get_width()//2, 90))

        y = 118
        line_h = 38

        def row(label, value_str, color=WHITE, bar_val=None, bar_col=GREEN):
            nonlocal y
            lbl = small_font.render(label, True, GREY)
            surface.blit(lbl, (12, y))
            val = font.render(value_str, True, color)
            surface.blit(val, (W - val.get_width() - 10, y))
            if bar_val is not None:
                self._gauge_bar(surface, 12, y + 17, W - 24, 6,
                                bar_val, bar_col)
                y += line_h
            else:
                y += line_h - 10

        row("MAX SPEED",
            f"{self.max_speed_seen:.1f} km/h", YELLOW)
        row("THROTTLE",
            f"{self.vehicle.throttle * 100:.0f}%",
            GREEN, self.vehicle.throttle, GREEN)
        row("BRAKE",
            f"{self.vehicle.brake_input * 100:.0f}%",
            RED, self.vehicle.brake_input, RED)
        steer_deg = math.degrees(self.vehicle.steering)
        row("STEERING",
            f"{steer_deg:+.1f} deg",
            CYAN,
            (self.vehicle.steering / TrackVehicleState.MAX_STEER + 1) * 0.5,
            CYAN)
        yaw_dps = math.degrees(self.vehicle.yaw_rate)
        row("YAW RATE",
            f"{yaw_dps:+.1f} d/s", ORANGE)
        lat_g = self.vehicle.lateral_accel / 9.81
        row("LATERAL G",
            f"{lat_g:+.2f} G",
            ORANGE,
            min(1.0, abs(lat_g) / 2.0),
            ORANGE)
        long_g = self.vehicle.long_accel / 9.81
        row("LONG G",
            f"{long_g:+.2f} G",
            BLUE,
            min(1.0, (long_g + 1.5) / 3.0),
            BLUE)

        pygame.draw.line(surface, (40, 55, 70), (8, y), (W - 8, y), 1)
        y += 6

        lat_err_m = self.lateral_error / 50.0   # px -> metres
        row("LAT ERROR",
            f"{lat_err_m:+.2f} m",
            RED if abs(lat_err_m) > 0.5 else WHITE)
        row("PROGRESS",
            f"{self.progress * 100:.1f}%", PURPLE)

        # V0.6: Delta timer
        delta = self._delta_at_progress(self.progress)
        if delta is not None:
            dcol = GREEN if delta > 0 else RED
            ds = font.render(
                ("AHEAD " if delta > 0 else "BEHIND") +
                f" {abs(delta):.3f}s",
                True, dcol
            )
            surface.blit(ds, (W//2 - ds.get_width()//2, y))
            y += 26

        pygame.draw.line(surface, (40, 55, 70), (8, y), (W - 8, y), 1)
        y += 6

        # V0.6: G-force circle
        circle_r = min(55, (HEIGHT - y - 120) // 2)
        if circle_r > 20:
            circle_cx = W // 2
            circle_cy = y + circle_r + 8
            self._draw_gforce_circle(surface, circle_cx, circle_cy, circle_r)
            glbl = small_font.render("FRICTION CIRCLE", True, (70, 80, 95))
            surface.blit(glbl, (W//2 - glbl.get_width()//2,
                                circle_cy + circle_r + 4))
            y = circle_cy + circle_r + 14

        # 4-Wheel Tyre Dynamic Load Transfer Indicator
        pygame.draw.line(surface, (40, 55, 70), (8, y), (W - 8, y), 1)
        y += 5
        t_lbl = small_font.render("TYRE LOAD TRANSFER", True, (130, 155, 185))
        surface.blit(t_lbl, (W//2 - t_lbl.get_width()//2, y))
        y += 16

        veh = self.vehicle
        t_loads = [veh.fl_load, veh.fr_load, veh.rl_load, veh.rr_load]
        max_t_load = max(max(t_loads), 800.0)

        def t_col(ld):
            rat = ld / max_t_load
            return RED if rat > 0.85 else (ORANGE if rat > 0.65 else GREEN)

        ch_w, ch_h = 70, 42
        ch_cx = W // 2
        ch_cy = y + ch_h // 2
        pygame.draw.rect(surface, (22, 28, 38), (ch_cx - ch_w//2, ch_cy - ch_h//2, ch_w, ch_h), border_radius=4)
        pygame.draw.rect(surface, (45, 56, 74), (ch_cx - ch_w//2, ch_cy - ch_h//2, ch_w, ch_h), 1, border_radius=4)
        # Front nose tick
        pygame.draw.polygon(surface, CYAN, [(ch_cx, ch_cy - ch_h//2 - 4), (ch_cx - 4, ch_cy - ch_h//2), (ch_cx + 4, ch_cy - ch_h//2)])

        # FL tyre & load
        fl_c = t_col(veh.fl_load)
        pygame.draw.rect(surface, fl_c, (ch_cx - ch_w//2 - 13, ch_cy - ch_h//2, 9, 17), border_radius=2)
        surface.blit(small_font.render(f"FL {int(veh.fl_load)}N", True, fl_c), (10, ch_cy - ch_h//2))

        # FR tyre & load
        fr_c = t_col(veh.fr_load)
        pygame.draw.rect(surface, fr_c, (ch_cx + ch_w//2 + 4, ch_cy - ch_h//2, 9, 17), border_radius=2)
        fr_s = small_font.render(f"{int(veh.fr_load)}N FR", True, fr_c)
        surface.blit(fr_s, (W - 10 - fr_s.get_width(), ch_cy - ch_h//2))

        # RL tyre & load
        rl_c = t_col(veh.rl_load)
        pygame.draw.rect(surface, rl_c, (ch_cx - ch_w//2 - 13, ch_cy + ch_h//2 - 17, 9, 17), border_radius=2)
        surface.blit(small_font.render(f"RL {int(veh.rl_load)}N", True, rl_c), (10, ch_cy + ch_h//2 - 15))

        # RR tyre & load
        rr_c = t_col(veh.rr_load)
        pygame.draw.rect(surface, rr_c, (ch_cx + ch_w//2 + 4, ch_cy + ch_h//2 - 17, 9, 17), border_radius=2)
        rr_s = small_font.render(f"{int(veh.rr_load)}N RR", True, rr_c)
        surface.blit(rr_s, (W - 10 - rr_s.get_width(), ch_cy + ch_h//2 - 15))

        y = ch_cy + ch_h // 2 + 10

        # Off-track warning
        if self.off_track:
            warn = medium_font.render("!! OFF TRACK !!", True, RED)
            surface.blit(warn, (W//2 - warn.get_width()//2, y))
            y += 28

        # Controls & Navigation reminder
        controls = [
            "W/S  Throttle / Brake",
            "A/D  Steer",
            "C    Camera (3D/FPV/2D)",
            "TAB  Wide / Split View",
            "L    Design Lab",
            "J    Suspension Lab",
            "T    Telemetry Graphs",
            "H    Handling Test",
            "ESC  Drive Screen",
        ]
        y = HEIGHT - len(controls) * 16 - 6
        for ctrl in controls:
            s2 = small_font.render(ctrl, True, (80, 92, 108))
            surface.blit(s2, (12, y))
            y += 18

    def _draw_right_panel(self, surface):
        """Lap timer, sector times, minimap — full right column."""
        RW = self.RIGHT_PANEL_W
        rx = WIDTH - RW
        if self.camera_mode == 2:
            rect = pygame.Rect(rx, 0, RW, HEIGHT)
            pygame.draw.rect(surface, (16, 20, 28), rect)
            pygame.draw.rect(surface, (40, 50, 65), rect, 1)
        else:
            panel_surf = pygame.Surface((RW, HEIGHT), pygame.SRCALPHA)
            panel_surf.fill((12, 16, 24, 215))
            surface.blit(panel_surf, (rx, 0))
            pygame.draw.line(surface, (45, 60, 85), (rx, 0), (rx, HEIGHT), 1)

        lt = self.lap_timer
        y = 12

        # Lap number
        s = medium_font.render(f"LAP  {lt.lap_number}", True, WHITE)
        surface.blit(s, (rx + RW//2 - s.get_width()//2, y))
        y += 32

        pygame.draw.line(surface, (40, 55, 70),
                         (rx + 8, y), (WIDTH - 8, y), 1)
        y += 8

        # Current lap time (large)
        cur_t = LapTimer.format_time(lt.current_lap_time)
        s = big_font.render(cur_t, True, CYAN)
        surface.blit(s, (rx + RW//2 - s.get_width()//2, y))
        y += 42

        # Best lap
        best_t = LapTimer.format_time(lt.best_lap_time)
        lbl = small_font.render("BEST", True, GREY)
        val = font.render(best_t, True, YELLOW)
        surface.blit(lbl, (rx + 12, y))
        surface.blit(val, (WIDTH - val.get_width() - 10, y))
        y += 30

        pygame.draw.line(surface, (40, 55, 70),
                         (rx + 8, y), (WIDTH - 8, y), 1)
        y += 10

        # Sector number
        s = font.render(f"SECTOR  {lt.current_sector}", True, ORANGE)
        surface.blit(s, (rx + RW//2 - s.get_width()//2, y))
        y += 30

        # Sector times
        sec_labels = ["S1", "S2", "S3"]
        sec_colors = [YELLOW, ORANGE, PURPLE]
        for si in range(3):
            is_current = (lt.current_sector == si + 1)
            t_str = LapTimer.format_time(lt.last_sector_times[si])
            best_str = LapTimer.format_time(lt.best_sector_times[si])
            col = sec_colors[si]
            # Label + current sector time
            lbl = font.render(
                sec_labels[si] + "  " + t_str, True,
                col if is_current else GREY
            )
            surface.blit(lbl, (rx + 12, y))
            # Best
            b_surf = small_font.render("B " + best_str, True,
                                       (100, 140, 100))
            surface.blit(b_surf, (rx + 12, y + 18))
            y += 48

        pygame.draw.line(surface, (40, 55, 70),
                         (rx + 8, y), (WIDTH - 8, y), 1)
        y += 10

        # Minimap — takes up the rest of the right column
        # (NuclearHUD panels are drawn on top of the right column by NuclearHUD.draw)
        map_rect = pygame.Rect(
            rx + 10, y, RW - 20,
            min(self.MAP_SIZE, HEIGHT - y - 10)
        )
        if map_rect.height > 50:
            self._draw_minimap(surface, map_rect)
        # Return the y position after minimap for HUD stacking
        self._right_panel_hud_y = map_rect.y + map_rect.height + 8

    def draw(self, surface):
        """Draw the entire Track Simulator screen."""
        surface.fill(BG)

        # View rectangle: full screen for 3D modes, center column for 2D top-down
        if self.camera_mode != 2 or self.wide_mode:
            view_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
        else:
            view_rect = pygame.Rect(
                self.LEFT_PANEL_W, 0,
                WIDTH - self.LEFT_PANEL_W - self.RIGHT_PANEL_W, HEIGHT
            )

        # Dispatch main view based on camera mode
        if self.camera_mode == 2:
            self._draw_track_view(surface)
        else:
            self._draw_3d_view(surface, view_rect)

        # Draw telemetry panels and title bar if not in wide mode
        if not self.wide_mode:
            self._draw_left_panel(surface)
            self._draw_right_panel(surface)

            # Title bar with live camera mode status
            mode_str = "3D CHASE" if self.camera_mode == 0 else ("3D COCKPIT" if self.camera_mode == 1 else "2D TOP-DOWN")
            title_surf = medium_font.render(
                f"TRACK SIMULATOR  |  C=Cam [{mode_str}]  TAB=Wide  V=Aero  N=Neural  E=Engineer  G=Ghost  P=Heatmap  F=CSV  ESC=Back",
                True, (70, 80, 100)
            )
            view_cx = self.LEFT_PANEL_W + (
                WIDTH - self.LEFT_PANEL_W - self.RIGHT_PANEL_W
            ) // 2
            title_x = view_cx - title_surf.get_width() // 2
            title_x = max(self.LEFT_PANEL_W + 4, title_x)
            surface.blit(title_surf, (title_x, 6))
        else:
            # Floating minimal HUD for wide immersive mode
            self._draw_wide_hud(surface)

        # V0.6: CSV export notice
        if self._export_notice > 0:
            fname = os.path.basename(self._export_filename)
            notice = font.render(
                f"EXPORTED  {fname}", True, GREEN
            )
            nx = WIDTH // 2 - notice.get_width() // 2
            ny = HEIGHT - 34
            pygame.draw.rect(surface, (14, 20, 14),
                             (nx - 8, ny - 4,
                              notice.get_width() + 16,
                              notice.get_height() + 8),
                             border_radius=5)
            surface.blit(notice, (nx, ny))


# ============================================================
# V1.0  MODULE 1: AERODYNAMICS SIMULATION
# ============================================================
# Full downforce/drag map vs speed, aero balance, DRS zones,
# wake turbulence behind ghost car.
# ============================================================

class AeroSimulation:
    """
    Computes and tracks aerodynamic forces in real time.

    Downforce model:
        F_df = 0.5 * rho * Cd_L * A * v^2  (simplified)
        Mapped to a tunable 'aero_balance' (% front downforce).

    Drag model:
        F_drag = 0.5 * rho * Cd * A * v^2

    DRS zones: pre-defined progress ranges on the straight.
        When active, Cd_drag is reduced by 15%.

    Wake turbulence:
        When within 40 px of the ghost car AND in the ghost's wake
        cone (heading-aligned), downforce is reduced by 20% and
        drag by 8% (dirty air effect).
    """

    RHO         = 1.225       # kg/m³ air density
    REF_AREA    = 1.0         # m² reference area
    CL_BASE     = 2.8         # downforce coefficient
    CD_BASE     = 1.15        # drag coefficient (total)
    DRS_CD_RED  = 0.15        # DRS drag reduction fraction
    WAKE_DF_PEN = 0.20        # downforce loss in wake
    WAKE_DR_REL = 0.08        # drag relief in wake
    WAKE_DIST   = 55          # px distance to trigger wake effect

    # DRS activation zones (track progress range, straight)
    DRS_ZONES = [
        (0.00, 0.14),   # start/finish straight
    ]

    def __init__(self, p):
        self.p = p
        # Tunable aero balance (% of downforce on front axle)
        self.aero_balance = 0.48     # 48% front
        self.drs_active    = False
        self.in_wake       = False
        self.downforce     = 0.0     # N total
        self.drag          = 0.0     # N
        self.front_df      = 0.0
        self.rear_df       = 0.0
        # History for animated HUD
        self.df_history  = []
        self.drag_history = []
        self._max_history = 300

    def _drs_active(self, progress):
        for lo, hi in self.DRS_ZONES:
            if lo <= progress <= hi:
                return True
        return False

    def _in_wake(self, vx, vy, ghost):
        """Return True if vehicle is inside the ghost car's turbulent wake."""
        if not ghost.enabled or not ghost.has_ghost or not ghost._ghost:
            return False
        idx = min(ghost._frame, len(ghost._ghost) - 1)
        gx, gy, gh, _ = ghost._ghost[idx]
        dx = vx - gx
        dy = vy - gy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > self.WAKE_DIST:
            return False
        # Check if vehicle is BEHIND ghost (in wake cone ±30 deg)
        wake_angle = math.atan2(dy, dx)
        angle_diff = abs(math.atan2(
            math.sin(wake_angle - gh),
            math.cos(wake_angle - gh)
        ))
        return angle_diff < math.radians(30)

    def update(self, speed_ms, progress, vx, vy, ghost):
        """
        speed_ms : vehicle speed in m/s
        progress : normalised track progress [0,1]
        vx, vy   : vehicle screen position
        ghost    : GhostCar instance
        """
        self.drs_active = self._drs_active(progress)
        self.in_wake    = self._in_wake(vx, vy, ghost)

        cd = self.CD_BASE * (1.0 - (self.DRS_CD_RED if self.drs_active else 0.0))
        cl = self.CL_BASE * (1.0 - (self.WAKE_DF_PEN if self.in_wake else 0.0))
        if self.in_wake:
            cd *= (1.0 - self.WAKE_DR_REL)

        q = 0.5 * self.RHO * self.REF_AREA * speed_ms * speed_ms
        self.downforce = cl * q
        self.drag      = cd * q
        self.front_df  = self.downforce * self.aero_balance
        self.rear_df   = self.downforce * (1.0 - self.aero_balance)

        self.df_history.append(self.downforce)
        self.drag_history.append(self.drag)
        if len(self.df_history) > self._max_history:
            self.df_history.pop(0)
            self.drag_history.pop(0)

    def draw_hud(self, surface, x, y, w, h):
        """
        Render the Aero HUD panel at (x,y) with size (w,h).
        Shows: downforce bar, drag bar, aero balance arc,
               DRS status, wake indicator, mini history graph.
        """
        # Panel background
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        panel_surf.fill((12, 18, 30, 210))
        pygame.draw.rect(panel_surf, (40, 200, 255, 80),
                         (0, 0, w, h), 1, border_radius=10)
        surface.blit(panel_surf, (x, y))

        title = medium_font.render("AERODYNAMICS", True, (40, 200, 255))
        surface.blit(title, (x + w//2 - title.get_width()//2, y + 8))

        # --- DRS indicator ---
        drs_col = GREEN if self.drs_active else (60, 70, 85)
        drs_surf = font.render(
            "DRS OPEN" if self.drs_active else "DRS CLOSED", True, drs_col
        )
        surface.blit(drs_surf, (x + 10, y + 36))

        # --- Wake indicator ---
        wake_col = (255, 160, 40) if self.in_wake else (40, 50, 65)
        wake_surf = small_font.render(
            "WAKE TURBULENCE" if self.in_wake else "CLEAN AIR",
            True, wake_col
        )
        surface.blit(wake_surf, (x + w - wake_surf.get_width() - 8, y + 39))

        # --- Downforce bar ---
        bar_y = y + 66
        bar_w = w - 20
        max_df = max(1.0, max(self.df_history) if self.df_history else 1.0)
        df_frac = min(1.0, self.downforce / max_df)
        lbl = small_font.render(f"DOWNFORCE  {self.downforce:.0f} N", True, CYAN)
        surface.blit(lbl, (x + 10, bar_y))
        pygame.draw.rect(surface, (18, 28, 42), (x+10, bar_y+18, bar_w, 10), border_radius=4)
        pygame.draw.rect(surface, CYAN,
                         (x+10, bar_y+18, int(df_frac*bar_w), 10), border_radius=4)

        # --- Drag bar ---
        bar_y2 = bar_y + 42
        max_drag = max(1.0, max(self.drag_history) if self.drag_history else 1.0)
        drag_frac = min(1.0, self.drag / max_drag)
        lbl2 = small_font.render(f"DRAG       {self.drag:.0f} N", True, ORANGE)
        surface.blit(lbl2, (x + 10, bar_y2))
        pygame.draw.rect(surface, (18, 28, 42), (x+10, bar_y2+18, bar_w, 10), border_radius=4)
        pygame.draw.rect(surface, ORANGE,
                         (x+10, bar_y2+18, int(drag_frac*bar_w), 10), border_radius=4)

        # --- Aero balance arc / dial ---
        arc_cx = x + w // 2
        arc_cy = y + 195
        arc_r  = 44
        pygame.draw.circle(surface, (18, 28, 42), (arc_cx, arc_cy), arc_r)
        pygame.draw.circle(surface, (40, 55, 75), (arc_cx, arc_cy), arc_r, 1)
        # Balance needle (0%=left, 100%=right)
        needle_angle = math.pi + self.aero_balance * math.pi  # 180->360 deg
        nx_end = arc_cx + int(math.cos(needle_angle) * (arc_r - 6))
        ny_end = arc_cy + int(math.sin(needle_angle) * (arc_r - 6))
        pygame.draw.line(surface, (40, 200, 255),
                         (arc_cx, arc_cy), (nx_end, ny_end), 3)
        pygame.draw.circle(surface, (40, 200, 255), (arc_cx, arc_cy), 4)
        bal_lbl = small_font.render(
            f"AERO BAL  {self.aero_balance*100:.0f}% F", True, WHITE
        )
        surface.blit(bal_lbl, (arc_cx - bal_lbl.get_width()//2, arc_cy + arc_r + 4))
        # Front/rear labels
        fl = small_font.render(f"F {self.front_df:.0f}N", True, CYAN)
        rl = small_font.render(f"R {self.rear_df:.0f}N", True, GREEN)
        surface.blit(fl, (x+10, arc_cy + arc_r + 4))
        surface.blit(rl, (x + w - rl.get_width() - 8, arc_cy + arc_r + 4))

        # --- Mini downforce history graph ---
        gh_y = y + 268
        gh_h = 50
        graph_lbl = small_font.render("DF HISTORY", True, (60, 70, 90))
        surface.blit(graph_lbl, (x+10, gh_y - 15))
        if len(self.df_history) >= 2:
            max_d = max(self.df_history) or 1.0
            gpts = []
            for i, d in enumerate(self.df_history):
                gx2 = x + 10 + int(i / (len(self.df_history)-1) * bar_w)
                gy2 = gh_y + gh_h - int(d / max_d * gh_h)
                gpts.append((gx2, gy2))
            if len(gpts) >= 2:
                pygame.draw.lines(surface, (40, 200, 255), False, gpts, 2)


# ============================================================
# V1.0  MODULE 2: AI RACE ENGINEER
# ============================================================
# Expert system that monitors live telemetry and generates
# text diagnosis + setup recommendations every N seconds.
# ============================================================

class AIRaceEngineer:
    """
    The AI Race Engineer analyses a rolling window of telemetry
    and fires human-readable diagnoses with concrete setup tips.

    Detection rules (priority ordered):
      1. Oversteer  : rear slip >> front slip, yaw_rate spike
      2. Understeer : front slip >> rear, high lat_g, steering saturated
      3. Late Apex  : high lateral error on exit of a corner
      4. Tyre Overload : any wheel load > 90% of safe max
      5. Off-throttle instability : yaw spike at throttle release
      6. Braking instability : yaw during heavy brake
      7. Aero wake : in turbulent air region
      8. Optimal   : none of the above — positive feedback
    """

    ANALYSIS_INTERVAL = 4.0   # seconds between new diagnoses
    WINDOW = 240              # frames of rolling history

    # Severity colours
    SEV_OK      = (80, 220, 130)
    SEV_INFO    = (65, 210, 230)
    SEV_WARN    = (245, 205, 80)
    SEV_ALERT   = (255, 150, 70)
    SEV_CRIT    = (240, 80, 80)

    def __init__(self, params):
        self.p = params
        self._timer  = 0.0
        self.messages = []    # list of (text, colour, age)
        self._MAX_MSGS = 6
        # Rolling telemetry buffers
        self._lat_g   = []
        self._long_g  = []
        self._yaw     = []    # deg/s
        self._steer   = []    # deg
        self._speed   = []    # km/h
        self._throttle= []
        self._lat_err = []    # m
        self._fl_load = []
        self._fr_load = []
        self._rl_load = []
        self._rr_load = []
        self._in_wake = []
        # Animated typewriter state
        self._typing_msg  = ""
        self._typed_chars = 0
        self._typing_col  = self.SEV_OK
        self._type_speed  = 40   # chars/sec
        self._type_accum  = 0.0

    def feed(self, lat_g, long_g, yaw_dps, steer_deg, speed_kmh,
             throttle, lat_err_m, fl, fr, rl, rr, in_wake):
        """Feed one frame of telemetry into the rolling buffers."""
        def push(buf, val):
            buf.append(val)
            if len(buf) > self.WINDOW:
                buf.pop(0)
        push(self._lat_g,    lat_g)
        push(self._long_g,   long_g)
        push(self._yaw,      yaw_dps)
        push(self._steer,    steer_deg)
        push(self._speed,    speed_kmh)
        push(self._throttle, throttle)
        push(self._lat_err,  lat_err_m)
        push(self._fl_load,  fl)
        push(self._fr_load,  fr)
        push(self._rl_load,  rl)
        push(self._rr_load,  rr)
        push(self._in_wake,  1.0 if in_wake else 0.0)

    def _avg(self, buf):
        return sum(buf) / len(buf) if buf else 0.0

    def _max_abs(self, buf):
        return max(abs(v) for v in buf) if buf else 0.0

    def _analyse(self):
        """Run the expert-system rules and return (message, colour)."""
        if len(self._lat_g) < 30:
            return ("Collecting data...", self.SEV_INFO)

        avg_lat   = self._avg([abs(v) for v in self._lat_g])
        max_yaw   = self._max_abs(self._yaw)
        avg_steer = self._avg([abs(v) for v in self._steer])
        avg_speed = self._avg(self._speed)
        avg_thr   = self._avg(self._throttle)
        max_lat_err = self._max_abs(self._lat_err)
        wake_frac = self._avg(self._in_wake)

        # Load stats
        all_loads = self._fl_load + self._fr_load + self._rl_load + self._rr_load
        if all_loads:
            max_load = max(all_loads)
            avg_fl = self._avg(self._fl_load)
            avg_fr = self._avg(self._fr_load)
            avg_rl = self._avg(self._rl_load)
            avg_rr = self._avg(self._rr_load)
            front_load = avg_fl + avg_fr
            rear_load  = avg_rl + avg_rr
            load_bias  = (front_load - rear_load) / max(1.0, front_load + rear_load)
        else:
            max_load = 0
            load_bias = 0.0

        # Tyre max capacity
        static_load = self.p.mass * 9.81 / 4.0
        overload_thresh = static_load * 1.85

        # ---- RULE 1: Oversteer ----
        # High yaw rate + low lateral accel = snap oversteer
        if max_yaw > 65 and avg_lat < 0.55:
            return (
                "OVERSTEER DETECTED  |  Rear losing grip — "
                f"increase rear ARB by ~150 N·m/rad or soften rear spring",
                self.SEV_CRIT
            )

        # ---- RULE 2: Understeer ----
        if avg_steer > 18 and avg_lat < 0.45 and avg_speed > 20:
            return (
                "UNDERSTEER  |  Front washing out — "
                "try reducing front ARB by 100 N·m/rad or lowering front roll centre",
                self.SEV_ALERT
            )

        # ---- RULE 3: Braking instability ----
        brake_frames = [b for b in self._throttle if b < 0.05]
        yaw_under_brake = self._max_abs(self._yaw[-60:]) if len(self._yaw) >= 60 else 0.0
        if yaw_under_brake > 50 and len(brake_frames) > 20:
            return (
                "BRAKING INSTABILITY  |  Yaw during heavy brake — "
                "increase front brake bias slightly or check rear damper",
                self.SEV_CRIT
            )

        # ---- RULE 4: Aero wake ----
        if wake_frac > 0.25:
            return (
                "DIRTY AIR  |  In ghost car wake — "
                "downforce -20% · expect understeer · create separation",
                self.SEV_WARN
            )

        # ---- RULE 5: Tyre overload ----
        if max_load > overload_thresh:
            corner = "FL" if max(self._fl_load, default=0) > overload_thresh else \
                     "FR" if max(self._fr_load, default=0) > overload_thresh else \
                     "RL" if max(self._rl_load, default=0) > overload_thresh else "RR"
            return (
                f"TYRE OVERLOAD  {corner}  |  Peak load critical — "
                "increase spring rate or reduce aero balance",
                self.SEV_CRIT
            )

        # ---- RULE 6: Front-heavy handling ----
        if load_bias > 0.08:
            return (
                "FRONT-HEAVY SETUP  |  High longitudinal load transfer — "
                "raise rear spring rate or increase rear aero balance",
                self.SEV_WARN
            )

        # ---- RULE 7: High lateral error ----
        if max_lat_err > 1.8:
            return (
                "TRACK POSITIONING  |  Large lateral error — "
                "use more track width on entry, hit apex earlier",
                self.SEV_INFO
            )

        # ---- RULE 8: Low speed ----
        if avg_speed < 15 and avg_thr < 0.3:
            return (
                "THROTTLE APPLICATION  |  Underdriving — "
                "apply throttle earlier on exit to maximise straight-line speed",
                self.SEV_INFO
            )

        # ---- OPTIMAL ----
        phrases = [
            "Balance looks good — keep pushing the tyres to the limit",
            "Clean lap — tyre loads well distributed across all corners",
            "Setup balanced — no handling anomalies detected this window",
            "Aero working nicely — exploit DRS on the straight for more top speed",
        ]
        import random
        return (random.choice(phrases), self.SEV_OK)

    def update(self, dt):
        """Should be called every frame. Returns True if new diagnosis."""
        self._timer -= dt
        # Advance typewriter
        if self._typed_chars < len(self._typing_msg):
            self._type_accum += dt
            chars_to_add = int(self._type_accum * self._type_speed)
            if chars_to_add > 0:
                self._typed_chars = min(
                    self._typed_chars + chars_to_add,
                    len(self._typing_msg)
                )
                self._type_accum = 0.0

        if self._timer > 0:
            return False

        self._timer = self.ANALYSIS_INTERVAL
        msg, col = self._analyse()
        self._typing_msg  = msg
        self._typed_chars = 0
        self._type_accum  = 0.0
        self._typing_col  = col
        return True

    def draw(self, surface, x, y, w, h):
        """
        Render the AI Race Engineer panel.
        Glassmorphism dark panel with animated typewriter output.
        """
        # Background
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((8, 14, 24, 220))
        pygame.draw.rect(bg, (*self._typing_col, 100),
                         (0, 0, w, h), 1, border_radius=10)
        surface.blit(bg, (x, y))

        # Title row
        title = medium_font.render("AI RACE ENGINEER", True, (180, 100, 255))
        surface.blit(title, (x + w//2 - title.get_width()//2, y + 8))

        # Animated blinking AI dot
        t_now = pygame.time.get_ticks() / 1000.0
        dot_col = (180, 100, 255) if int(t_now * 2) % 2 == 0 else (60, 30, 100)
        pygame.draw.circle(surface, dot_col, (x + 18, y + 16), 5)

        # Separator
        pygame.draw.line(surface, (60, 30, 120),
                         (x+10, y+33), (x+w-10, y+33), 1)

        # Typewriter message
        visible = self._typing_msg[:self._typed_chars]
        # Word-wrap at word boundaries to fit within w
        words = visible.split(" ")
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if small_font.size(test)[0] <= w - 20:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        ty = y + 40
        for line in lines:
            surf = small_font.render(line, True, self._typing_col)
            surface.blit(surf, (x + 10, ty))
            ty += 18
            if ty > y + h - 12:
                break

        # Blinking cursor
        if self._typed_chars < len(self._typing_msg) or int(t_now * 3) % 2 == 0:
            cur_surf = small_font.render("_", True, self._typing_col)
            surface.blit(cur_surf, (x + 10 + small_font.size(
                lines[-1] if lines else "")[0], ty - 18))


# ============================================================
# V1.0  MODULE 3: NEURAL LAP PREDICTOR
# ============================================================
# Online linear regression trained on telemetry in real time.
# Predicts finishing lap time from partial lap data.
# Outputs: predicted time, confidence interval, time-loss map.
# ============================================================

class NeuralLapPredictor:
    """
    A streaming online regression engine (incremental least-squares)
    that fits lap time = f(speed_profile, lateral_g, throttle)
    using the features captured so far in the current lap.

    Architecture:
      Feature vector per progress sample:
        [speed_kmh, abs(lat_g), long_g, throttle, brake,
         abs(yaw_dps), progress]
      Target: completed lap times from previous laps.

    The predictor uses Recursive Least Squares (RLS) — a classic
    online learning algorithm.  For each completed lap the model
    is updated.  The current-lap feature mean is then projected
    through the weights to estimate remaining time.

    Time-loss heatmap:
      Each centerline index is scored: if the vehicle is slower
      here vs the best lap at this progress, the index is marked.
    """

    N_FEATURES = 7
    FORGETTING  = 0.97    # RLS forgetting factor (λ)
    INIT_COVAR  = 1e4     # initial covariance diagonal

    def __init__(self, track_n):
        # RLS state
        self._weights = [0.0] * self.N_FEATURES   # regression weights
        self._P = [[self.INIT_COVAR if i == j else 0.0
                    for j in range(self.N_FEATURES)]
                   for i in range(self.N_FEATURES)]  # covariance matrix
        self._fitted    = False
        self._n_laps    = 0
        # Per-lap recording
        self._lap_data  = []    # [(features_vec, progress), ...]
        # Predictions
        self.predicted_time = None    # seconds
        self.confidence     = 0.0     # 0-1
        self.lap_times      = []      # completed lap times for display
        # Per-index speed comparison vs best lap
        self._best_speed    = [0.0] * track_n
        self._cur_speed     = [0.0] * track_n
        self.time_loss_map  = [0.0] * track_n   # positive = losing time here

    def _features(self, speed_kmh, lat_g, long_g, throttle,
                  brake, yaw_dps, progress):
        return [
            speed_kmh / 80.0,
            abs(lat_g),
            long_g,
            throttle,
            brake,
            abs(yaw_dps) / 180.0,
            progress
        ]

    def _dot(self, a, b):
        return sum(x*y for x, y in zip(a, b))

    def _mat_vec(self, M, v):
        """Matrix-vector multiply M @ v."""
        return [self._dot(row, v) for row in M]

    def _outer(self, u, v):
        """Outer product u @ v^T."""
        return [[u[i]*v[j] for j in range(len(v))] for i in range(len(u))]

    def _rls_update(self, x, y):
        """Update weights with one (x, y) training sample via RLS."""
        lam = self.FORGETTING
        # Gain vector: K = P x / (lam + x^T P x)
        Px = self._mat_vec(self._P, x)
        denom = lam + self._dot(x, Px)
        if abs(denom) < 1e-10:
            return
        K = [v / denom for v in Px]
        # Error
        err = y - self._dot(self._weights, x)
        # Weight update
        self._weights = [
            w + K[i] * err
            for i, w in enumerate(self._weights)
        ]
        # Covariance update: P = (P - K x^T P) / lam
        KxP = [[K[i] * Px[j] for j in range(self.N_FEATURES)]
               for i in range(self.N_FEATURES)]
        self._P = [
            [(self._P[i][j] - KxP[i][j]) / lam
             for j in range(self.N_FEATURES)]
            for i in range(self.N_FEATURES)
        ]

    def record_frame(self, idx, speed_kmh, lat_g, long_g,
                     throttle, brake, yaw_dps, progress):
        """Feed one frame of current-lap data."""
        fv = self._features(speed_kmh, lat_g, long_g,
                            throttle, brake, yaw_dps, progress)
        self._lap_data.append((fv, progress))
        # Speed comparison vs best
        self._cur_speed[idx] = speed_kmh
        if self._best_speed[idx] > 0:
            delta = self._best_speed[idx] - speed_kmh
            # Positive delta = slower here than best
            self.time_loss_map[idx] = max(-1.0, min(1.0, delta / 30.0))

    def commit_lap(self, lap_time, best_speed_snapshot=None):
        """
        Called when a lap completes. Update RLS with completed data.
        lap_time : float, actual completed lap time in seconds
        """
        if not self._lap_data:
            return
        self._n_laps += 1
        self.lap_times.append(lap_time)
        if len(self.lap_times) > 10:
            self.lap_times.pop(0)

        # Promote best speed
        for i, s in enumerate(self._cur_speed):
            if s > self._best_speed[i]:
                self._best_speed[i] = s

        # Build a single training example from lap feature mean + actual time
        if self._lap_data:
            means = [0.0] * self.N_FEATURES
            for fv, _ in self._lap_data:
                for j, val in enumerate(fv):
                    means[j] += val
            n = len(self._lap_data)
            means = [v / n for v in means]
            self._rls_update(means, lap_time)
            self._fitted = True

        self._lap_data = []
        self._cur_speed = [0.0] * len(self._cur_speed)

    def predict(self):
        """
        Predict finishing lap time using partial current-lap data.
        Returns (predicted_seconds, confidence_0_to_1) or (None, 0).
        """
        if not self._fitted or len(self._lap_data) < 20:
            return None, 0.0

        # Mean features so far
        means = [0.0] * self.N_FEATURES
        for fv, _ in self._lap_data:
            for j, val in enumerate(fv):
                means[j] += val
        n = len(self._lap_data)
        means = [v / n for v in means]
        pred = self._dot(self._weights, means)

        # Confidence grows with number of laps and progress
        progress_frac = self._lap_data[-1][1] if self._lap_data else 0.0
        confidence = min(1.0, (self._n_laps / 5.0) * progress_frac)
        return pred, confidence

    def draw(self, surface, x, y, w, h, track, cam_x, cam_y):
        """
        Draw the Neural Predictor HUD panel AND the time-loss heatmap
        overlay on the track.
        """
        # --- Panel ---
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 16, 28, 215))
        pygame.draw.rect(bg, (255, 200, 50, 90), (0, 0, w, h), 1, border_radius=10)
        surface.blit(bg, (x, y))

        title = medium_font.render("NEURAL LAP PREDICTOR", True, (245, 205, 80))
        surface.blit(title, (x + w//2 - title.get_width()//2, y + 8))

        pygame.draw.line(surface, (100, 85, 20),
                         (x+10, y+33), (x+w-10, y+33), 1)

        pred, conf = self.predict()
        ty = y + 40

        if pred is not None:
            pred_str = LapTimer.format_time(max(0, pred))
            p_surf = big_font.render(pred_str, True, (245, 205, 80))
            surface.blit(p_surf, (x + w//2 - p_surf.get_width()//2, ty))
            ty += 40

            # Confidence bar
            clbl = small_font.render(f"CONFIDENCE  {conf*100:.0f}%", True, WHITE)
            surface.blit(clbl, (x+10, ty))
            ty += 18
            bar_w = w - 20
            pygame.draw.rect(surface, (30, 35, 45), (x+10, ty, bar_w, 8), border_radius=3)
            conf_col = GREEN if conf > 0.6 else YELLOW if conf > 0.3 else RED
            pygame.draw.rect(surface, conf_col,
                             (x+10, ty, int(conf*bar_w), 8), border_radius=3)
            ty += 20
        else:
            msg = small_font.render(
                "Complete 1 lap to activate", True, (90, 100, 120)
            )
            surface.blit(msg, (x + w//2 - msg.get_width()//2, ty))
            ty += 22

        # Lap time history sparkline
        if len(self.lap_times) >= 2:
            spark_lbl = small_font.render("LAP HISTORY", True, (80, 90, 110))
            surface.blit(spark_lbl, (x+10, ty))
            ty += 16
            t_min = min(self.lap_times)
            t_max = max(self.lap_times) + 0.1
            bar_w = w - 20
            spark_h = 32
            pts = []
            for i, lt in enumerate(self.lap_times):
                sx2 = x + 10 + int(i / (len(self.lap_times)-1) * bar_w)
                sy2 = ty + spark_h - int((lt - t_min) / (t_max - t_min) * spark_h)
                pts.append((sx2, sy2))
            if len(pts) >= 2:
                pygame.draw.lines(surface, (245, 205, 80), False, pts, 2)
            # Best lap label
            best_str = LapTimer.format_time(t_min)
            bl = small_font.render(f"BEST {best_str}", True, GREEN)
            surface.blit(bl, (x + w//2 - bl.get_width()//2, ty + spark_h + 2))

        # --- Time-loss heatmap on track ---
        # Overlay: green = faster than best, red = slower
        if self._n_laps >= 1:
            cl = track.centerline
            n = len(cl)
            for i in range(n - 1):
                loss = self.time_loss_map[i]
                if abs(loss) < 0.05:
                    continue
                # colour: red for losing time, green for gaining
                if loss > 0:
                    r = min(255, int(loss * 400))
                    col = (r, 30, 30, 180)
                else:
                    g = min(255, int(-loss * 400))
                    col = (30, g, 30, 180)
                sx2 = int(cl[i].x - cam_x)
                sy2 = int(cl[i].y - cam_y)
                ex2 = int(cl[i+1].x - cam_x)
                ey2 = int(cl[i+1].y - cam_y)
                try:
                    pygame.draw.line(surface, col[:3], (sx2, sy2), (ex2, ey2), 5)
                except Exception:
                    pass


# ============================================================
# V1.0  MODULE 4: NUCLEAR HUD OVERLAY
# ============================================================
# Renders all V1.0 overlays into the Track Simulator screen.
# Manages aero HUD, AI engineer panel, neural predictor panel.
# Activated with N (neural), E (engineer), A (aero).
# ============================================================

class NuclearHUD:
    """
    Container that owns all V1.0 overlay systems and draws them
    as floating glassmorphism panels over the track view.
    Each can be toggled independently:
      N = Neural Lap Predictor
      E = AI Race Engineer
      A = Aerodynamics HUD
    """

    PANEL_W = 300
    PANEL_H_AERO = 335
    PANEL_H_AI   = 145
    PANEL_H_PRED = 195

    def __init__(self, params, track_n):
        self.aero  = AeroSimulation(params)
        self.ai    = AIRaceEngineer(params)
        self.pred  = NeuralLapPredictor(track_n)
        self.show_aero  = True
        self.show_ai    = True
        self.show_pred  = True
        self._lap_number_prev = 1

    def update(self, dt, vehicle, progress, ghost,
               lat_g, long_g, yaw_dps, steer_deg,
               speed_kmh, throttle, brake, lat_err_m,
               fl, fr, rl, rr, track_idx):
        """Update all V1.0 systems."""
        speed_ms = vehicle.speed
        self.aero.update(speed_ms, progress, vehicle.x, vehicle.y, ghost)
        self.ai.feed(
            lat_g, long_g, yaw_dps, steer_deg, speed_kmh,
            throttle, lat_err_m, fl, fr, rl, rr, self.aero.in_wake
        )
        self.ai.update(dt)
        self.pred.record_frame(
            track_idx, speed_kmh, lat_g, long_g,
            throttle, brake, yaw_dps, progress
        )

    def on_lap_complete(self, lap_time):
        """Call when a lap completes."""
        self.pred.commit_lap(lap_time)

    def handle_key(self, key):
        """Toggle overlays."""
        if key == pygame.K_n:
            self.show_pred = not self.show_pred
        elif key == pygame.K_e:
            self.show_ai   = not self.show_ai
        elif key == pygame.K_v:
            self.show_aero = not self.show_aero

    def draw(self, surface, track, cam_x, cam_y,
             left_panel_w, right_panel_w):
        """Draw all enabled overlays — stacked inside the right column."""
        # Right column: occupies x from (WIDTH - right_panel_w) to WIDTH
        rx = WIDTH - right_panel_w + 10
        rw = right_panel_w - 20

        # Start drawing below the minimap (set by TrackSimulator._draw_right_panel)
        from_y = getattr(
            getattr(self, '_ts_ref', None), '_right_panel_hud_y',
            HEIGHT // 2
        )

        y = from_y
        pad = 6

        # --- AERO HUD ---
        if self.show_aero and y + 40 < HEIGHT - 4:
            ph = min(self.PANEL_H_AERO, HEIGHT - y - 4)
            self.aero.draw_hud(surface, rx, y, rw, ph)
            y += ph + pad

        # --- NEURAL PREDICTOR ---
        if self.show_pred and y + 40 < HEIGHT - 4:
            ph = min(self.PANEL_H_PRED, HEIGHT - y - 4)
            self.pred.draw(
                surface, rx, y, rw, ph,
                track, cam_x, cam_y
            )
            y += ph + pad

        # --- AI ENGINEER ---
        if self.show_ai and y + 40 < HEIGHT - 4:
            ph = min(self.PANEL_H_AI, HEIGHT - y - 4)
            self.ai.draw(surface, rx, y, rw, ph)
            y += ph + pad

        # --- Toggle hints at bottom of right column ---
        hints = "V=Aero  N=Predict  E=Engineer"
        hs = small_font.render(hints, True, (50, 60, 80))
        hx = WIDTH - right_panel_w + (right_panel_w - hs.get_width()) // 2
        surface.blit(hs, (hx, HEIGHT - 20))


# ------------------------------------------------------------
# INSTANTIATE TRACK SIMULATOR
# ------------------------------------------------------------

track_simulator = TrackSimulator(params)


# ------------------------------------------------------------
# INSTANTIATE NUCLEAR HUD (V1.0)
# ------------------------------------------------------------

nuclear_hud = NuclearHUD(params, track_simulator.track.n)
# Give nuclear_hud a reference to the track simulator so it can read
# _right_panel_hud_y (set each frame by _draw_right_panel) and place
# HUD panels directly below the minimap without overlapping the track.
nuclear_hud._ts_ref = track_simulator

# ============================================================
# MAIN STATE
# ============================================================

mode = "DRIVE"

running = (__name__ == "__main__")


# ============================================================
# MAIN LOOP
# ============================================================

while running:

    dt = (
        clock.tick(60) /
        1000.0
    )

    dt = min(
        dt,
        0.03
    )

    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        if event.type == pygame.KEYDOWN:

            # =================================================
            # DRIVE
            # =================================================

            if mode == "DRIVE":

                if event.key == pygame.K_l:

                    mode = "DESIGN"

                elif event.key == pygame.K_j:

                    mode = "SUSPENSION"

                elif event.key == pygame.K_h:

                    # H = HANDLING TEST
                    handling_test.run()
                    mode = "HANDLING"

                elif event.key == pygame.K_t:

                    mode = "TELEMETRY"

                elif event.key == pygame.K_k:

                    # K = TRACK SIMULATOR
                    track_simulator = TrackSimulator(params)
                    mode = "TRACK"

                elif event.key == pygame.K_r:

                    reset_all()

            # =================================================
            # TELEMETRY
            # =================================================

            elif mode == "TELEMETRY":

                if event.key in (
                    pygame.K_t,
                    pygame.K_ESCAPE
                ):

                    mode = "DRIVE"

            # =================================================
            # HANDLING
            # =================================================

            elif mode == "HANDLING":

                if event.key in (
                    pygame.K_RETURN,
                    pygame.K_ESCAPE
                ):

                    mode = "DRIVE"

            # =================================================
            # DESIGN
            # =================================================

            elif mode == "DESIGN":

                if event.key == pygame.K_ESCAPE:

                    mode = "DRIVE"

                elif event.key == pygame.K_UP:

                    design_selected = (
                        design_selected -
                        1
                    ) % len(
                        design_parameters
                    )

                elif event.key == pygame.K_DOWN:

                    design_selected = (
                        design_selected +
                        1
                    ) % len(
                        design_parameters
                    )

                elif event.key == pygame.K_LEFT:

                    modify_design(
                        design_parameters[
                            design_selected
                        ],
                        -1
                    )

                elif event.key == pygame.K_RIGHT:

                    modify_design(
                        design_parameters[
                            design_selected
                        ],
                        1
                    )

                elif event.key == pygame.K_RETURN:

                    mode = "DRIVE"

                elif event.key == pygame.K_SPACE:

                    sweep_parameter = (
                        design_selected
                    )

                    run_sensitivity()

                    mode = "SWEEP"

                elif event.key == pygame.K_r:

                    params.__init__()

                    vehicle.create_tires()
                    vehicle.create_suspension()
                    vehicle.reset()

            # =================================================
            # SWEEP
            # =================================================

            elif mode == "SWEEP":

                if event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_RETURN
                ):

                    mode = "DESIGN"

            # =================================================
            # SUSPENSION
            # =================================================

            elif mode == "SUSPENSION":

                if event.key == pygame.K_ESCAPE:

                    mode = "DRIVE"

                elif event.key == pygame.K_UP:

                    suspension_selected = (
                        suspension_selected -
                        1
                    ) % len(
                        suspension_parameters
                    )

                elif event.key == pygame.K_DOWN:

                    suspension_selected = (
                        suspension_selected +
                        1
                    ) % len(
                        suspension_parameters
                    )

                elif event.key == pygame.K_LEFT:

                    modify_suspension(
                        suspension_parameters[
                            suspension_selected
                        ],
                        -1
                    )

                elif event.key == pygame.K_RIGHT:

                    modify_suspension(
                        suspension_parameters[
                            suspension_selected
                        ],
                        1
                    )

                elif event.key == pygame.K_RETURN:

                    mode = "DRIVE"

                elif event.key == pygame.K_r:

                    params.__init__()

                    vehicle.create_tires()
                    vehicle.create_suspension()
                    vehicle.reset()

            # =================================================
            # TRACK SIMULATOR
            # =================================================

            elif mode == "TRACK":

                if event.key == pygame.K_ESCAPE:

                    mode = "DRIVE"

                elif event.key == pygame.K_r:

                    track_simulator.reset_lap()

                elif event.key == pygame.K_g:

                    # G = toggle ghost car on/off
                    track_simulator.ghost.enabled = (
                        not track_simulator.ghost.enabled
                    )

                elif event.key == pygame.K_p:

                    # P = toggle speed heatmap on/off
                    track_simulator.show_heatmap = (
                        not track_simulator.show_heatmap
                    )

                elif event.key == pygame.K_f:

                    # F = export lap telemetry to CSV
                    fname = track_simulator.exporter.export()
                    if fname:
                        track_simulator._export_filename = fname
                        track_simulator._export_notice = 4.0

                elif event.key == pygame.K_c:

                    # C = cycle camera view (3D Chase -> 3D Cockpit FPV -> 2D Tactical)
                    track_simulator.cycle_camera()

                elif event.key == pygame.K_TAB:

                    # TAB = toggle full-screen wide mode vs split telemetry panels
                    track_simulator.toggle_wide_mode()

                elif event.key == pygame.K_l:

                    # L = Design Lab
                    mode = "DESIGN"

                elif event.key == pygame.K_j:

                    # J = Suspension Lab
                    mode = "SUSPENSION"

                elif event.key == pygame.K_t:

                    # T = Telemetry Graphs
                    mode = "TELEMETRY"

                elif event.key == pygame.K_h:

                    # H = Handling Test
                    handling_test.run()
                    mode = "HANDLING"

                # V1.0 Nuclear HUD toggles
                else:
                    nuclear_hud.handle_key(event.key)

    # ========================================================
    # DRIVE PHYSICS
    # ========================================================

    if mode == "DRIVE":

        keys = pygame.key.get_pressed()

        vehicle.throttle = (
            1.0
            if keys[pygame.K_w]
            else 0.0
        )

        vehicle.brake = (
            1.0
            if keys[pygame.K_s]
            else 0.0
        )

        steering = 0

        if keys[pygame.K_a]:
            steering -= 1

        if keys[pygame.K_d]:
            steering += 1

        vehicle.steering = (
            steering *
            math.radians(30)
        )

        vehicle.update(
            dt
        )

        record_telemetry()

    # ========================================================
    # DRAW
    # ========================================================

    if mode == "DRIVE":

        draw_drive()

    elif mode == "TELEMETRY":

        draw_telemetry()

    elif mode == "HANDLING":

        draw_handling_result()

    elif mode == "DESIGN":

        draw_design_lab()

    elif mode == "SWEEP":

        draw_sweep_result()

    elif mode == "SUSPENSION":

        draw_suspension_lab()

    elif mode == "TRACK":

        # Track simulator physics update
        keys = pygame.key.get_pressed()
        ts_throttle = 1.0 if keys[pygame.K_w] else 0.0
        ts_brake    = 1.0 if keys[pygame.K_s] else 0.0
        ts_steer    = 0
        if keys[pygame.K_a]:
            ts_steer -= 1
        if keys[pygame.K_d]:
            ts_steer += 1

        # --- V1.0: detect lap completion BEFORE update for predictor ---
        _prev_lap = track_simulator.lap_timer.lap_number

        track_simulator.update(dt, ts_throttle, ts_brake, ts_steer)

        # --- V1.0: feed nuclear HUD systems ---
        ts = track_simulator
        veh = ts.vehicle
        lat_g  = veh.lateral_accel / 9.81
        long_g = veh.long_accel    / 9.81
        yaw_d  = math.degrees(veh.yaw_rate)
        steer_d = math.degrees(veh.steering)
        spd_k  = veh.speed * 3.6
        # Use live track vehicle's wheel loads for AI engineer
        _fl = veh.fl_load
        _fr = veh.fr_load
        _rl = veh.rl_load
        _rr = veh.rr_load
        nuclear_hud.update(
            dt, veh, ts.progress, ts.ghost,
            lat_g, long_g, yaw_d, steer_d,
            spd_k, ts_throttle, ts_brake,
            ts.lateral_error / 50.0,
            _fl, _fr, _rl, _rr,
            ts.closest_idx
        )

        # Notify predictor on lap complete
        if ts.lap_timer.lap_number != _prev_lap:
            _lt = ts.lap_timer.best_lap_time
            if _lt is not None:
                nuclear_hud.on_lap_complete(_lt)

        track_simulator.draw(screen)

        # --- V1.0: draw nuclear HUD overlays on top (when in panel mode) ---
        if not ts.wide_mode:
            nuclear_hud.draw(
                screen,
                ts.track,
                ts.cam_x, ts.cam_y,
                ts.LEFT_PANEL_W,
                ts.RIGHT_PANEL_W
            )

    pygame.display.flip()


if __name__ == "__main__":
    pygame.quit()
