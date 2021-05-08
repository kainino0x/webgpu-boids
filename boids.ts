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

  const NUM_BOIDS = 1500;

  // **************************************************************************
  // Device and canvas initialization
  // **************************************************************************

  assert('gpu' in navigator, 'WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter !== null, 'requestAdapter failed');
  const gpu = await adapter.requestDevice();

  const cvs = document.getElementById('cvs') as HTMLCanvasElement;
  cvs.width = WIDTH;
  cvs.height = HEIGHT;
  const ctx = cvs.getContext('gpupresent');
  assert(ctx !== null, 'Unable to create gpupresent context');
  const swapChain = ctx.configureSwapChain({ device: gpu, format: SWAP_CHAIN_FORMAT });

  // **************************************************************************
  // Shaders
  // **************************************************************************

  const computeShaderModule = gpu.createShaderModule({
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

        for (var i: u32 = 0u; i < arrayLength(particlesA.particles); i = i + 1u) {
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
      }
      `,
  });

  const renderShaderModule = gpu.createShaderModule({
    code: `

      // **********************************************************************
      // Rendering
      // **********************************************************************

      let boid_vertices: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
          vec2<f32>(-0.01, -0.02),
          vec2<f32>( 0.01, -0.02),
          vec2<f32>( 0.00,  0.02));

      [[stage(vertex)]]
      fn renderBoids_vert(
          [[builtin(vertex_index)]] VertexIndex: u32,
          [[location(0)]] a_particlePos: vec2<f32>,
          [[location(1)]] a_particleVel: vec2<f32>
        ) -> [[builtin(position)]] vec4<f32> {
        let angle: f32 = -atan2(a_particleVel.x, a_particleVel.y);
        let boid_vertex: vec2<f32> = boid_vertices[VertexIndex];
        // TODO: change rotation into a matrix
        let pos: vec2<f32> = vec2<f32>(
            (boid_vertex.x * cos(angle)) - (boid_vertex.y * sin(angle)),
            (boid_vertex.x * sin(angle)) + (boid_vertex.y * cos(angle)));
        return vec4<f32>(pos + a_particlePos, 0.0, 1.0);
      }

      [[stage(fragment)]]
      fn renderBoids_frag() -> [[location(0)]] vec4<f32> {
        // TODO: lighting?
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
      }`,
  })

  // **************************************************************************
  // Compute pipeline setup
  // **************************************************************************

  const stepBoidsSimulation_pipeline = gpu.createComputePipeline({
    compute: {
      module: computeShaderModule,
      entryPoint: 'stepBoidsSimulation'
    }
  });

  // **************************************************************************
  // Render pipeline setup
  // **************************************************************************

  const renderBoids_pipeline = gpu.createRenderPipeline({
    vertex: {
      module: renderShaderModule,
      entryPoint: 'renderBoids_vert',
      buffers: [
        {
          // instanced particles buffer
          arrayStride: 4 * 4,
          stepMode: 'instance',
          attributes: [
            // instance position
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            // instance velocity
            { shaderLocation: 1, offset: 2 * 4, format: 'float32x2' },
          ],
        },
      ],
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: 'renderBoids_frag',
      targets: [
        { format: SWAP_CHAIN_FORMAT },
      ],
    },
    multisample: {
      count: 4,
    },
    depthStencil: {
      format: DEPTH_FORMAT,
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // **************************************************************************
  // Resources setup
  // **************************************************************************

  const simParamBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT;
  const simParamBuffer = gpu.createBuffer({
    mappedAtCreation: true,
    size: simParamBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  new Float32Array(simParamBuffer.getMappedRange()).set([
    0.04,  // deltaT
    0.1,   // rule1Distance
    0.025, // rule2Distance
    0.025, // rule3Distance
    0.02,  // rule1Scale
    0.05,  // rule2Scale
    0.005, // rule3Scale
  ]);
  simParamBuffer.unmap();

  const initialParticleData = new Float32Array(NUM_BOIDS * 4);
  for (let i = 0; i < NUM_BOIDS; ++i) {
    initialParticleData[4 * i + 0] = 2 * (Math.random() - 0.5);
    initialParticleData[4 * i + 1] = 2 * (Math.random() - 0.5);
    initialParticleData[4 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
    initialParticleData[4 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
  }

  const particleBuffers: GPUBuffer[] = new Array(2);
  const particleBindGroups: GPUBindGroup[] = new Array(2);
  for (let i = 0; i < 2; ++i) {
    particleBuffers[i] = gpu.createBuffer({
      size: initialParticleData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(particleBuffers[i].getMappedRange()).set(
      initialParticleData
    );
    particleBuffers[i].unmap();
  }

  for (let i = 0; i < 2; ++i) {
    particleBindGroups[i] = gpu.createBindGroup({
      layout: stepBoidsSimulation_pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: simParamBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: particleBuffers[i],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: particleBuffers[(i + 1) % 2],
            offset: 0,
            size: initialParticleData.byteLength,
          },
        },
      ],
    });
  }

  // **************************************************************************
  // Render pass setup
  // **************************************************************************

  // Create a multisampled color texture as "scratch space" for the render pass colors
  const multisampleColorTexture = gpu.createTexture({
    size: [WIDTH, HEIGHT],
    format: SWAP_CHAIN_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: 4,
  });
  const multisampleColorTextureView = multisampleColorTexture.createView();

  // Create a multisampled depth texture as "scratch space" for the render pass depth values
  const depthTexture = gpu.createTexture({
    size: [WIDTH, HEIGHT],
    format: DEPTH_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: 4,
  });
  const depthTextureView = depthTexture.createView();

  const renderPassDescriptor = {
    colorAttachments: [{
      // Use multisampleColorTextureView as "scratch space" for multisampled rendering
      view: multisampleColorTextureView,
      // Load a constant color (dark blue) at the beginning of the render pass
      loadValue: [0.1, 0.0, 0.5, 1.0],
      // Resolve multisampled rendering into the canvas texture (to be set later)
      resolveTarget: null! as GPUTextureView,
      // Multisampled rendering results can be discarded after resolve
      storeOp: 'clear' as const,
    }],
    depthStencilAttachment: {
      // Use depthTextureView as "scratch space"
      view: depthTextureView,
      // Load a constant value (1) at the beginning of the render pass
      depthLoadValue: 1,
      // Depth-testing buffer can be discarded after pass
      depthStoreOp: 'clear' as const,
      // (Not used but required)
      stencilLoadValue: 0,
      stencilStoreOp: 'clear' as const,
    }
  };

  // **************************************************************************
  // Render loop
  // **************************************************************************

  let frameNum = 0;

  function stepBoidsSimulation(commandEncoder: GPUCommandEncoder) {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(stepBoidsSimulation_pipeline);
    // Simulate either from particleBuffers[0] -> particleBuffers[1] or vice versa
    passEncoder.setBindGroup(0, particleBindGroups[frameNum % 2]);
    passEncoder.dispatch(Math.ceil(NUM_BOIDS / 64));
    passEncoder.endPass();
  }

  function renderBoids(commandEncoder: GPUCommandEncoder) {
    renderPassDescriptor.colorAttachments[0].resolveTarget = swapChain.getCurrentTexture().createView();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderBoids_pipeline);
    // Render from the particleBuffers[x] that was just updated
    passEncoder.setVertexBuffer(0, particleBuffers[(frameNum + 1) % 2]);
    passEncoder.draw(3, NUM_BOIDS);
    passEncoder.endPass();
  }

  function frame() {
    stats.begin();

    const commandEncoder = gpu.createCommandEncoder();
    {
      stepBoidsSimulation(commandEncoder);
      renderBoids(commandEncoder);
    }
    const commandBuffer = commandEncoder.finish();
    gpu.queue.submit([commandBuffer]);
    frameNum++;

    stats.end();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

function assert(condition: boolean, msg: string): asserts condition {
  if (!condition) {
    document.getElementById('log')!.textContent = msg;
    throw new Error(msg);
  }
}
