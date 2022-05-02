def ParticleFilter(play_ground, measurement, other=None):
    """
    Estimate the next (x,y) position of the satelite.
    This is the function you will have to write for part A.

    :param play_ground: Self Designed Class
        A model of the positions, velocities, and masses
        of the cars in the play_ground.
    :param measurement: float
        A floating point number representing
        the measured dis of landmarks or other cars in the playground
        felt at the target satellite at that point in time.
    :param other: any
        This is initially None, but if you return an OTHER from
        this function call, it will be passed back to you the next time it is
        called, so that you can use it to keep track of important information
        over time. (We suggest you use a dictionary so that you can store as many
        different named values as you want.)
    :return:
        estimate: Tuple[float, float]. The (x,y) estimate of the target satellite at the next timestep
        other: any. Any additional information you'd like to pass between invocations of this function
        optional_points_to_plot: List[Tuple[float, float, float]].
            A list of tuples like (x,y,h) to plot for the visualization
    """
    xy_estimate = (0.85445, 0.3185)
    optional_points_to_plot = []
    # Owen code:
    test_mode = False
    if other == None:
        ini_filter = False
        N = 500
        particles = []
        sigma = 0.09
        iter_time = 1
    else:
        ini_filter,N,particles,sigma,iter_time = other
        iter_time += 1
    """initialization"""
    if not ini_filter:
        for i in range(N):
            x = random.uniform(-4,4)
            y = random.uniform(-4,4)
            if x<1e-5 and y<1e-5:
                x+=1e-3
                y+=1e-3
            r = [x*AU,y*AU]
            particle = Body.create_body_at_xy(r=r, mass=0, measurement_noise=0, play_ground=play_ground)
            particles.append(particle)
        if test_mode:
            particles[-1] = Body.create_body_at_xy(r=[0.65*AU, 0.65*AU], mass=0, measurement_noise=0, play_ground=play_ground)
        ini_filter = True

    """Weights"""
    Weights = []
    mu = measurement
    s = sigma*mu
    for particle in particles:
        v = particle.measure(planets=play_ground.planets)
        prob = exp(- ((mu - v) ** 2) / (s ** 2) / 2.0) / sqrt(2.0 * pi * (s ** 2))
        Weights.append(prob)
    Weight_sum = sum(Weights)
    
    """Resample"""
    new_particles = []
    new_Weights = []
    beta = 0
    max_Weight = max(Weights)
    index = random.randint(0,N-1)
    for i in range(N):
        beta += random.uniform(0, 2.0*max_Weight)
        while beta > Weights[index]:
            beta -= Weights[index]
            index = (index + 1) % N
        r = particles[index].r
        v = particles[index].v
        new_particles.append(Body(r=r, v=v, mass=0, measurement_noise=0))
        new_Weights.append(Weights[index])
       
    """Fuzz"""
    percentage = 0.6
    fuzz_sigma = 0.05*AU*(exp(1/iter_time))
    fuzz_size = int(N*percentage)
    for i in range(fuzz_size):
        index = random.randint(0, N-1)
        rx,ry = new_particles[index].r
        vx,vy = new_particles[index].v
        new_particles[index] = Body(r=[random.gauss(rx,fuzz_sigma), random.gauss(ry,fuzz_sigma)], v=[random.gauss(vx,0),random.gauss(vy,0)], mass=0, measurement_noise=0)
    particles = new_particles
     
    """Mimic"""
    for index,particle in enumerate(particles):
        particles[index] = play_ground.move_body(particle)
        
    """Estimate"""
    x_sum = 0
    y_sum = 0
    for index,particle in enumerate(particles):
        x_sum += particle.r[0]
        y_sum += particle.r[1]
    xy_estimate = (x_sum/N, y_sum/N)   
    other = [ini_filter,N,particles,sigma,iter_time]
    for i in range(N):
        x,y = particles[i].r
        vx,vy = particles[i].v
        h = atan2(vy,vx)
        optional_point_to_plot = (x,y,h)
        optional_points_to_plot.append(optional_point_to_plot)
    return xy_estimate, other, optional_points_to_plot
