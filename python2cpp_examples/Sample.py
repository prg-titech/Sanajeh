class Body:  # クラスをDynaSOArを使う必要があることを何らかのシンタックスで宣言すべき（DynaSOArだとAllocator::Baseの子クラスにする）

    # ここでAllocatorのFieldを呼び出す----------------------------------------------------------------------------------
    # declare_field_types(Body, float, float, float, float, float, float, float);
    # Field<Body, 0> pos_x_;
    # Field<Body, 1> pos_y_;
    # Field<Body, 2> vel_x_;
    # Field<Body, 3> vel_y_;
    # Field<Body, 4> force_x_;
    # Field<Body, 5> force_y_;
    # Field<Body, 6> mass_;
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float
    # ---------------------------------------------------------------------------------------------------------------

    def __init__(self, px: float, py: float, vx: float, vy: float, m: float):
        self.pos_x = px
        self.pos_y = py
        self.vel_x = vx
        self.vel_y = vy
        self.mass = m
        self.force_x = 0.0
        self.force_y = 0.0

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        # ここでdevice_doを呼び出す-------------------------------------------------------------------------------------
        # device_allocator->template device_do<Body>(&Body::apply_force, this);
        __pyallocator__.device_do(Body, Body.apply_force, self)
        # -----------------------------------------------------------------------------------------------------------

    def apply_force(self, other: Body):
        if other is not self:
            dx: float = self.pos_x - other.pos_x
            dy: float = self.pos_x - other.pos_y
            dist: float = math.sqrt(dx * dx + dy * dy)
            f: float = kGravityConstant * self.mass * other.mass / (dist * dist + kDampeningFactor)

            other.force_x += f * dx / dist
            other.force_y += f * dy / dist

    def body_update(self):

        """
        test code
        """
        dy: int = kNumBodies

        self.vel_x += self.force_x * kDt / self.mass
        self.vel_y += self.force_y * kDt / self.mass
        self.pos_x += self.vel_x * kDt
        self.pos_y += self.vel_y * kDt

        if self.pos_x < -1 or self.pos_x > 1:
            self.vel_x = -self.vel_x

        if self.pos_y < -1 or self.pos_y > 1:
            self.vel_y = -self.vel_y

