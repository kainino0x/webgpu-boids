// `boids.js` is generated from `boids.ts`. Run `npm run build` to build it.

function assert(condition: boolean, msg: string): asserts condition {
  if (!condition) {
    const log = document.getElementById('log')!;
    if (!log.textContent) {
      log.textContent = msg;
    }
    throw new Error(msg);
  }
}

// Add stats.js
var stats = new Stats();
stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
document.body.appendChild(stats.dom);

// Based on http://austin-eng.com/webgpu-samples/samples/computeBoids

const WIDTH = 500 * window.devicePixelRatio;
const HEIGHT = WIDTH;
const CANVAS_FORMAT = 'bgra8unorm';
const DEPTH_FORMAT = 'depth24plus';

const NUM_BOIDS = 500;

// **************************************************************************
// Device and canvas initialization
// **************************************************************************

function setCanvasSize(canvas: HTMLCanvasElement, w: number, h: number) {
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
}

// Device is initialized without a canvas. Can be used with zero or more canvases.
assert('gpu' in navigator, 'WebGPU not supported');
const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();
assert(adapter !== null, 'requestAdapter failed');
const device: GPUDevice = await adapter.requestDevice();
device.onuncapturederror = (ev) => {
  paused = true;
  console.warn(ev.error);
  assert(false, ev.error.message);
};

// Canvas context is initialized without a device.
const canvas = document.getElementById('cvs') as HTMLCanvasElement;
setCanvasSize(canvas, WIDTH, HEIGHT);
const canvasContext: GPUCanvasContext = canvas.getContext('webgpu')!;
assert(canvasContext !== null, 'Unable to create "webgpu" canvas context');

// Configure the canvas context to associate it with a device and set its format.
canvasContext.configure({ device, format: CANVAS_FORMAT });

// **************************************************************************
// Compute Shader
// **************************************************************************

const computeShaderModule: GPUShaderModule = device.createShaderModule({
  code: /* wgsl */ `
    struct Particle {
      pos: vec2f,
      vel: vec2f,
    }
    struct SimParams {
      deltaT: f32,
      rule1Distance: f32,
      rule2Distance: f32,
      rule3Distance: f32,
      rule1Scale: f32,
      rule2Scale: f32,
      rule3Scale: f32,
    }
    @binding(0) @group(0) var<uniform> params: SimParams;
    @binding(1) @group(0) var<storage, read> particlesA: array<Particle>;
    @binding(2) @group(0) var<storage, read_write> particlesB: array<Particle>;

    @compute @workgroup_size(64)
    fn stepBoidsSimulation(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
      let index = GlobalInvocationID.x;

      var vPos = particlesA[index].pos;
      var vVel = particlesA[index].vel;
      var cMass = vec2f(0.0, 0.0);
      var cVel = vec2f(0.0, 0.0);
      var colVel = vec2f(0.0, 0.0);
      var cMassCount = 0u;
      var cVelCount = 0u;
      var pos: vec2f;
      var vel: vec2f;

      for (var i = 0u; i < ${NUM_BOIDS}u; i = i + 1u) {
        if i == index {
          continue;
        }

        pos = particlesA[i].pos.xy;
        vel = particlesA[i].vel.xy;
        if distance(pos, vPos) < params.rule1Distance {
          cMass = cMass + pos;
          cMassCount = cMassCount + 1u;
        }
        if distance(pos, vPos) < params.rule2Distance {
          colVel = colVel - (pos - vPos);
        }
        if distance(pos, vPos) < params.rule3Distance {
          cVel = cVel + vel;
          cVelCount = cVelCount + 1u;
        }
      }
      if cMassCount > 0u {
        let temp = f32(cMassCount);
        cMass = (cMass / vec2f(temp, temp)) - vPos;
      }
      if cVelCount > 0u {
        let temp = f32(cVelCount);
        cVel = cVel / vec2f(temp, temp);
      }
      vVel = vVel + (cMass * params.rule1Scale) + (colVel * params.rule2Scale) +
          (cVel * params.rule3Scale);

      // clamp velocity for a more pleasing simulation
      vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
      // kinematic update
      vPos = vPos + (vVel * params.deltaT);
      // Wrap around boundary
      vPos = (vPos + 1.0) % 1.0;
      // Write back
      particlesB[index].pos = vPos;
      particlesB[index].vel = vVel;
    }`,
});

// **************************************************************************
// Compute pipeline setup
// **************************************************************************

const stepBoidsSimulation_pipeline: GPUComputePipeline =
  device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: computeShaderModule,
      entryPoint: 'stepBoidsSimulation', // Entry point to use as compute shader
    },
  });

// **************************************************************************
// Rendering Shaders
// **************************************************************************

// Shader module can define multiple entry points (here, vertex and fragment).
const renderShaderModule: GPUShaderModule = device.createShaderModule({
  code: /* wgsl */ `
    var<private> boid_positions: array<vec3f, 6> = array<vec3f, 6>(
        vec3f(-0.04, -0.05, 0.50), vec3f( 0.00, -0.04, 0.47), vec3f( 0.00,  0.04, 0.50),
        vec3f( 0.00, -0.04, 0.47), vec3f( 0.04, -0.05, 0.50), vec3f( 0.00,  0.04, 0.50)
      );
    var<private> boid_normals: array<vec3f, 2> = array<vec3f, 2>(
        vec3f(-0.6, 0.2, -1.0),
        vec3f( 0.6, 0.2, -1.0)
      );

    struct Varying {
      @builtin(position) pos: vec4f,
      @location(0) vtxpos: vec3f,
      @location(1) normal: vec3f,
      @location(2) @interpolate(flat) index: u32,
    }

    @vertex
    fn renderBoids_vert(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32,
        @location(0) a_particlePos: vec2f,
        @location(1) a_particleVel: vec2f
      ) -> Varying {
      let angle = -atan2(a_particleVel.x, a_particleVel.y);

      let boid_vtxpos = boid_positions[vertexIndex];
      let boid_normal = boid_normals[vertexIndex / 3u];

      let rotation = mat3x3f(
          vec3f(cos(angle), sin(angle), 0.0),
          vec3f(-sin(angle), cos(angle), 0.0),
          vec3f(0.0, 0.0, 1.0));

      // map simulation space (0,0 to 1,1) to NDC (-1,-1 to 1,1)
      let ndcPos = a_particlePos * 2.0 - 1.0;
      var vary: Varying;
      vary.vtxpos = rotation * boid_vtxpos + vec3f(ndcPos, 0.0);
      vary.pos = vec4f(vary.vtxpos, 1.0);
      vary.normal = rotation * boid_normal;
      vary.index = instanceIndex;
      return vary;
    }

    @fragment
    fn renderBoids_frag(vary: Varying) -> @location(0) vec4f {
      var color = vec4f(0.0, 0.0, 0.0, 1.0);
      {
        // Constant light position
        let lightPos = vec3f(0.5, -0.5, 0.3);
        let lightDir = lightPos - vary.vtxpos;
        let light = dot(normalize(vary.normal), normalize(lightDir)) / (length(lightDir) + 0.5);
        color.r = light;
      }
      {
        // Constant light position
        let lightPos = vec3f(-0.5, 0.5, 0.3);
        let lightDir = lightPos - vary.vtxpos;
        let light = dot(normalize(vary.normal), normalize(lightDir)) / (length(lightDir) + 0.5);
        color.g = light;
      }
      if vary.index % 2u == 1u {
        color.b = 1.0;
      }
      return color;
    }`,
});

// **************************************************************************
// Render pipeline setup
// **************************************************************************

const renderBoids_pipeline: GPURenderPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    // List of vertex buffers
    buffers: [
      // Layout of vertex buffer 0, the instanced particles buffer
      {
        arrayStride: 16,
        stepMode: 'instance',
        // List of attributes inside this vertex buffer
        attributes: [
          { format: 'float32x2', offset: 0, shaderLocation: 0 }, // Particle position
          { format: 'float32x2', offset: 8, shaderLocation: 1 }, // Particle velocity
        ],
      },
    ],
    module: renderShaderModule,
    entryPoint: 'renderBoids_vert', // Entry point to use as vertex shader
  },
  primitive: { topology: 'triangle-list' },
  depthStencil: {
    format: DEPTH_FORMAT,
    depthWriteEnabled: true,
    // Configure depth test
    depthCompare: 'less',
  },
  multisample: { count: 4 },
  fragment: {
    module: renderShaderModule,
    entryPoint: 'renderBoids_frag', // Entry point to use as fragment shader
    targets: [{ format: CANVAS_FORMAT }], // List of render attachments
  },
});

// **************************************************************************
// Resources setup
// **************************************************************************

const simParamBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT;

// Create uniform buffer for simulation parameters
const simParamBuffer: GPUBuffer = device.createBuffer({
  mappedAtCreation: true, // Start the buffer in the 'mapped' state, for initialization
  size: simParamBufferSize,
  usage: GPUBufferUsage.UNIFORM,
});
const mappedRange: ArrayBuffer = simParamBuffer.getMappedRange();
new Float32Array(mappedRange).set([
  0.02, // deltaT
  0.05, // rule1Distance
  0.05, // rule2Distance
  0.025, // rule3Distance
  0.05, // rule1Scale
  0.02, // rule2Scale
  0.005, // rule3Scale
]);
simParamBuffer.unmap(); // Unmap it (detaches the ArrayBuffer)

// Initialize particle buffers with random data
const initialParticleData = new Float32Array(NUM_BOIDS * 4);
for (let i = 0; i < NUM_BOIDS; ++i) {
  initialParticleData[4 * i + 0] = Math.random();
  initialParticleData[4 * i + 1] = Math.random();
  initialParticleData[4 * i + 2] = 2 * (Math.random() - 0.5) * 0.1;
  initialParticleData[4 * i + 3] = 2 * (Math.random() - 0.5) * 0.1;
}

const particleBuffers: GPUBuffer[] = new Array(2);
for (let i = 0; i < 2; ++i) {
  particleBuffers[i] = device.createBuffer({
    size: initialParticleData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(particleBuffers[i].getMappedRange()).set(
    initialParticleData
  );
  particleBuffers[i].unmap();
}

// Get bind group layout automatically generated from shader
const bindGroupLayout = stepBoidsSimulation_pipeline.getBindGroupLayout(0);

// Create two bind groups:
// one for stepping from particleBuffers[0] to [1],
// one for stepping from particleBuffers[1] to [0] (ping-pong).
const particleBindGroups: GPUBindGroup[] = new Array(2);
for (let i = 0; i < 2; ++i) {
  particleBindGroups[i] = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: simParamBuffer } },
      { binding: 1, resource: { buffer: particleBuffers[i] } },
      { binding: 2, resource: { buffer: particleBuffers[(i + 1) % 2] } },
    ],
  });
}

// **************************************************************************
// Render pass setup
// **************************************************************************

// Create a multisampled color texture for rendering.
// This just a "scratch space": only the multisample-resolve result will be kept.
const multisampleColorTexture: GPUTexture = device.createTexture({
  size: [WIDTH, HEIGHT],
  format: CANVAS_FORMAT,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
  sampleCount: 4,
});
const multisampleColorTextureView: GPUTextureView =
  multisampleColorTexture.createView();

// Create a multisampled depth texture for rendering.
// This is also a "scratch space", for depth testing inside the render pass.
const depthTexture: GPUTexture = device.createTexture({
  size: [WIDTH, HEIGHT],
  format: DEPTH_FORMAT,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
  sampleCount: 4,
});
const depthTextureView: GPUTextureView = depthTexture.createView();

const renderPassDescriptor = {
  colorAttachments: [
    {
      view: multisampleColorTextureView,
      // Load a constant color (dark blue) at the beginning of the render pass.
      loadOp: 'clear' as const,
      clearValue: [0.1, 0.0, 0.3, 1.0],
      // Resolve multisampled rendering into the canvas texture (to be set later).
      resolveTarget: null! as GPUTextureView,
      // Multisampled rendering results can be discarded after resolve.
      storeOp: 'discard' as const,
    },
  ],
  depthStencilAttachment: {
    view: depthTextureView,
    // Load a constant value (1) at the beginning of the render pass.
    depthLoadOp: 'clear' as const,
    depthClearValue: 1,
    // Depth-testing buffer can be discarded after the render pass.
    depthStoreOp: 'discard' as const,
  },
};

// **************************************************************************
// Render loop
// **************************************************************************

let frameNum = 0;

let paused = false;
canvas.addEventListener('click', () => {
  paused = !paused;
});

function stepBoidsSimulation(commandEncoder: GPUCommandEncoder) {
  const passEncoder: GPUComputePassEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(stepBoidsSimulation_pipeline);
  // Simulate either from particleBuffers[0] -> particleBuffers[1] or vice versa.
  passEncoder.setBindGroup(0, particleBindGroups[frameNum % 2]);
  passEncoder.dispatchWorkgroups(Math.ceil(NUM_BOIDS / 64));

  passEncoder.end();
}

function renderBoids(commandEncoder: GPUCommandEncoder) {
  // We get a new GPUTexture from the swap chain every frame.
  renderPassDescriptor.colorAttachments[0].resolveTarget = canvasContext
    .getCurrentTexture()
    .createView();

  const passEncoder: GPURenderPassEncoder =
    commandEncoder.beginRenderPass(renderPassDescriptor);

  passEncoder.setPipeline(renderBoids_pipeline);
  // Render from the particleBuffers[x] that was just updated.
  passEncoder.setVertexBuffer(0, particleBuffers[(frameNum + 1) % 2]);
  passEncoder.draw(6, NUM_BOIDS);

  passEncoder.end();
}

function frame() {
  stats.begin();

  if (!paused) {
    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();
    {
      stepBoidsSimulation(commandEncoder);
      renderBoids(commandEncoder);
    }
    const commandBuffer: GPUCommandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);
    frameNum++;
  }

  stats.end();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

export {};
