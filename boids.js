"use strict";
// `boids.js` is generated from `boids.ts`. Run `npm run build` to build it.
// Add stats.js
var stats = new Stats();
stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
document.body.appendChild(stats.dom);
// Based on http://austin-eng.com/webgpu-samples/samples/computeBoids
(async () => {
    const WIDTH = 800;
    const HEIGHT = 800;
    const SWAP_CHAIN_FORMAT = 'bgra8unorm';
    const DEPTH_FORMAT = 'depth24plus';
    const NUM_BOIDS = 500;
    // **************************************************************************
    // Device and canvas initialization
    // **************************************************************************
    // Device is initialized without a canvas. Can be used with zero or more canvases.
    assert('gpu' in navigator, 'WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    assert(adapter !== null, 'requestAdapter failed');
    const device = await adapter.requestDevice();
    device.onuncapturederror = ev => {
        console.warn(ev.error);
    };
    // Canvas context is initialized without a device
    const cvs = document.getElementById('cvs');
    cvs.width = WIDTH;
    cvs.height = HEIGHT;
    cvs.style.width = (WIDTH / window.devicePixelRatio) + 'px';
    cvs.style.width = (WIDTH / window.devicePixelRatio) + 'px';
    const ctx = cvs.getContext('gpupresent');
    assert(ctx !== null, 'Unable to create gpupresent context');
    // Pair them to create a "swap chain" to vend render target textures
    const swapChain = ctx.configureSwapChain({ device, format: SWAP_CHAIN_FORMAT });
    // **************************************************************************
    // Shaders
    // **************************************************************************
    const computeShaderModule = device.createShaderModule({
        code: `
      struct Particle {
        pos: vec2<f32>;
        vel: vec2<f32>;
      };
      [[block]] struct SimParams {
        deltaT: f32;
        rule1Distance: f32;
        rule2Distance: f32;
        rule3Distance: f32;
        rule1Scale: f32;
        rule2Scale: f32;
        rule3Scale: f32;
      };
      [[block]] struct Particles {
        particles: [[stride(16)]] array<Particle>;
      };
      [[binding(0), group(0)]] var<uniform> params: SimParams;
      [[binding(1), group(0)]] var<storage> particlesA: [[access(read)]] Particles;
      [[binding(2), group(0)]] var<storage> particlesB: [[access(read_write)]] Particles;

      [[stage(compute), workgroup_size(64)]]
      fn stepBoidsSimulation([[builtin(global_invocation_id)]] GlobalInvocationID: vec3<u32>) {
        let index: u32 = GlobalInvocationID.x;

        var vPos: vec2<f32> = particlesA.particles[index].pos;
        var vVel: vec2<f32> = particlesA.particles[index].vel;
        var cMass: vec2<f32> = vec2<f32>(0.0, 0.0);
        var cVel: vec2<f32> = vec2<f32>(0.0, 0.0);
        var colVel: vec2<f32> = vec2<f32>(0.0, 0.0);
        var cMassCount: u32 = 0u;
        var cVelCount: u32 = 0u;
        var pos: vec2<f32>;
        var vel: vec2<f32>;

        for (var i: u32 = 0u; i < ${NUM_BOIDS}u; i = i + 1u) {
          if (i == index) {
            continue;
          }

          pos = particlesA.particles[i].pos.xy;
          vel = particlesA.particles[i].vel.xy;
          if (distance(pos, vPos) < params.rule1Distance) {
            cMass = cMass + pos;
            cMassCount = cMassCount + 1u;
          }
          if (distance(pos, vPos) < params.rule2Distance) {
            colVel = colVel - (pos - vPos);
          }
          if (distance(pos, vPos) < params.rule3Distance) {
            cVel = cVel + vel;
            cVelCount = cVelCount + 1u;
          }
        }
        if (cMassCount > 0u) {
          let temp: f32 = f32(cMassCount);
          cMass = (cMass / vec2<f32>(temp, temp)) - vPos;
        }
        if (cVelCount > 0u) {
          let temp: f32 = f32(cVelCount);
          cVel = cVel / vec2<f32>(temp, temp);
        }
        vVel = vVel + (cMass * params.rule1Scale) + (colVel * params.rule2Scale) +
            (cVel * params.rule3Scale);

        // clamp velocity for a more pleasing simulation
        vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
        // kinematic update
        vPos = vPos + (vVel * params.deltaT);
        // Wrap around boundary
        if (vPos.x < -1.0) {
          vPos.x = 1.0;
        }
        if (vPos.x > 1.0) {
          vPos.x = -1.0;
        }
        if (vPos.y < -1.0) {
          vPos.y = 1.0;
        }
        if (vPos.y > 1.0) {
          vPos.y = -1.0;
        }
        // Write back
        particlesB.particles[index].pos = vPos;
        particlesB.particles[index].vel = vVel;
      }`,
    });
    if ('compilationInfo' in computeShaderModule) {
        computeShaderModule.compilationInfo().then(info => {
            if (info.messages.length) {
                console.log(info.messages);
            }
        });
    }
    // Shader module can define multiple entry points (here, vertex and fragment).
    const renderShaderModule = device.createShaderModule({
        code: `
      let boid_positions: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
          vec3<f32>(-0.04, -0.05, 0.50), vec3<f32>( 0.00, -0.04, 0.47), vec3<f32>( 0.00,  0.04, 0.50),
          vec3<f32>( 0.00, -0.04, 0.47), vec3<f32>( 0.04, -0.05, 0.50), vec3<f32>( 0.00,  0.04, 0.50)
        );
      let boid_normals: array<vec3<f32>, 2> = array<vec3<f32>, 2>(
          vec3<f32>(-0.6, 0.2, -1.0),
          vec3<f32>( 0.6, 0.2, -1.0)
        );

      struct Varying {
        [[builtin(position)]] pos: vec4<f32>;
        [[location(0)]] vtxpos: vec3<f32>;
        [[location(1)]] normal: vec3<f32>;
        [[location(2)]] index: u32;
      };

      [[stage(vertex)]]
      fn renderBoids_vert(
          [[builtin(vertex_index)]] VertexIndex: u32,
          [[builtin(instance_index)]] InstanceIndex: u32,
          [[location(0)]] a_particlePos: vec2<f32>,
          [[location(1)]] a_particleVel: vec2<f32>
        ) -> Varying {
        let angle: f32 = -atan2(a_particleVel.x, a_particleVel.y);

        let boid_vtxpos: vec3<f32> = boid_positions[VertexIndex];
        let boid_normal: vec3<f32> = boid_normals[VertexIndex / 3u];

        let rotation: mat3x3<f32> = mat3x3<f32>(
            vec3<f32>(cos(angle), sin(angle), 0.0),
            vec3<f32>(-sin(angle), cos(angle), 0.0),
            vec3<f32>(0.0, 0.0, 1.0));

        var varying: Varying;
        varying.vtxpos = rotation * boid_vtxpos + vec3<f32>(a_particlePos, 0.0);
        varying.pos = vec4<f32>(varying.vtxpos, 1.0);
        varying.normal = rotation * boid_normal;
        varying.index = InstanceIndex;
        return varying;
      }

      [[stage(fragment)]]
      fn renderBoids_frag(v: Varying) -> [[location(0)]] vec4<f32> {
        var color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        {
          // Constant light position
          let lightPos: vec3<f32> = vec3<f32>(0.5, -0.5, 0.3);
          let lightDir: vec3<f32> = lightPos - v.vtxpos;
          let light: f32 = dot(normalize(v.normal), normalize(lightDir)) / (length(lightDir) + 0.5);
          color.r = light;
        }
        {
          // Constant light position
          let lightPos: vec3<f32> = vec3<f32>(-0.5, 0.5, 0.3);
          let lightDir: vec3<f32> = lightPos - v.vtxpos;
          let light: f32 = dot(normalize(v.normal), normalize(lightDir)) / (length(lightDir) + 0.5);
          color.g = light;
        }
        if (v.index % 2u == 1u) {
          color.b = 1.0;
        }
        return color;
      }`,
    });
    if ('compilationInfo' in renderShaderModule) {
        renderShaderModule.compilationInfo().then(info => {
            if (info.messages.length) {
                console.log(info.messages);
            }
        });
    }
    // **************************************************************************
    // Compute pipeline setup
    // **************************************************************************
    const stepBoidsSimulation_pipeline = device.createComputePipeline({
        compute: {
            module: computeShaderModule,
            entryPoint: 'stepBoidsSimulation', // Which entry point to use as compute shader
        }
    });
    // **************************************************************************
    // Render pipeline setup
    // **************************************************************************
    const renderBoids_pipeline = device.createRenderPipeline({
        vertex: {
            module: renderShaderModule,
            entryPoint: 'renderBoids_vert',
            buffers: [
                {
                    arrayStride: 4 * 4,
                    stepMode: 'instance',
                    attributes: [
                        { format: 'float32x2', offset: 0, shaderLocation: 0 },
                        { format: 'float32x2', offset: 2 * 4, shaderLocation: 1 }, // particle velocity
                    ],
                },
            ],
        },
        primitive: { topology: 'triangle-list' },
        depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: true,
            depthCompare: 'less',
        },
        fragment: {
            module: renderShaderModule,
            entryPoint: 'renderBoids_frag',
            targets: [
                { format: SWAP_CHAIN_FORMAT },
            ],
        },
    });
    // **************************************************************************
    // Resources setup
    // **************************************************************************
    // Create uniform buffer for simulation parameters
    const simParamBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT;
    const simParamBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: simParamBufferSize,
        usage: GPUBufferUsage.UNIFORM,
    });
    const mappedRange = simParamBuffer.getMappedRange();
    new Float32Array(mappedRange).set([
        0.04,
        0.1,
        0.100,
        0.050,
        0.05,
        0.02,
        0.005, // rule3Scale
    ]);
    simParamBuffer.unmap(); // Unmap it (detaches the ArrayBuffer)
    // Initialize particle buffers with random data
    const initialParticleData = new Float32Array(NUM_BOIDS * 4);
    for (let i = 0; i < NUM_BOIDS; ++i) {
        initialParticleData[4 * i + 0] = 2 * (Math.random() - 0.5);
        initialParticleData[4 * i + 1] = 2 * (Math.random() - 0.5);
        initialParticleData[4 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
        initialParticleData[4 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
    }
    const particleBuffers = new Array(2);
    for (let i = 0; i < 2; ++i) {
        particleBuffers[i] = device.createBuffer({
            size: initialParticleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(particleBuffers[i].getMappedRange()).set(initialParticleData);
        particleBuffers[i].unmap();
    }
    // Get bind group layout automatically generated from shader
    const bindGroupLayout = stepBoidsSimulation_pipeline.getBindGroupLayout(0);
    // Create two bind groups, one for stepping from particleBuffers[0]
    // to [1] and one for stepping from [1] to [0] (ping-pong).
    const particleBindGroups = new Array(2);
    for (let i = 0; i < 2; ++i) {
        particleBindGroups[i] = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: simParamBuffer } },
                { binding: 1, resource: { buffer: particleBuffers[i] } },
                {
                    binding: 2, resource: { buffer: particleBuffers[(i + 1) % 2] },
                },
            ],
        });
    }
    // **************************************************************************
    // Render pass setup
    // **************************************************************************
    // Create a depth texture "scratch space", for depth testing inside the render pass.
    const depthTexture = device.createTexture({
        size: [WIDTH, HEIGHT],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const depthTextureView = depthTexture.createView();
    const renderPassDescriptor = {
        colorAttachments: [{
                view: null,
                // Load a constant color (dark purple) at the beginning of the render pass.
                loadValue: [0.1, 0.0, 0.3, 1.0],
                // Store results at the end of the render pass.
                storeOp: 'store',
            }],
        depthStencilAttachment: {
            view: depthTextureView,
            // Load a constant value (1) at the beginning of the render pass.
            depthLoadValue: 1,
            // Depth-testing buffer can be discarded after the render pass.
            depthStoreOp: 'clear',
            // (Stencil not used, but these settings are required.)
            stencilLoadValue: 0,
            stencilStoreOp: 'clear',
        }
    };
    // **************************************************************************
    // Render loop
    // **************************************************************************
    let frameNum = 0;
    let paused = false;
    cvs.addEventListener('click', () => {
        paused = !paused;
    });
    function stepBoidsSimulation(commandEncoder) {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(stepBoidsSimulation_pipeline);
        // Simulate either from particleBuffers[0] -> particleBuffers[1] or vice versa.
        passEncoder.setBindGroup(0, particleBindGroups[frameNum % 2]);
        passEncoder.dispatch(Math.ceil(NUM_BOIDS / 64));
        passEncoder.endPass();
    }
    function renderBoids(commandEncoder) {
        // We get a new GPUTexture from the swap chain every frame.
        renderPassDescriptor.colorAttachments[0].view = swapChain.getCurrentTexture().createView();
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(renderBoids_pipeline);
        // Render from the particleBuffers[x] that was just updated.
        passEncoder.setVertexBuffer(0, particleBuffers[(frameNum + 1) % 2]);
        passEncoder.draw(6, NUM_BOIDS);
        passEncoder.endPass();
    }
    function frame() {
        stats.begin();
        if (!paused) {
            const commandEncoder = device.createCommandEncoder();
            {
                stepBoidsSimulation(commandEncoder);
                renderBoids(commandEncoder);
            }
            const commandBuffer = commandEncoder.finish();
            device.queue.submit([commandBuffer]);
            frameNum++;
        }
        stats.end();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
})();
function assert(condition, msg) {
    if (!condition) {
        document.getElementById('log').textContent = msg;
        throw new Error(msg);
    }
}
