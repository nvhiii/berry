import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass";
import { GUI } from "dat.gui";

const FluidBackground = () => {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const sceneRef = useRef(null);
  const materialRef = useRef(null);
  const meshRef = useRef(null);
  const rendererRef = useRef(null);
  const composerRef = useRef(null);
  const elasticComposerRef = useRef(null);
  const fluidSimRef = useRef(null);
  const mainUniformsRef = useRef(null);

  // --- Fluid Simulation ---
  const config = {
    SIM_RESOLUTION: 128,
    DYE_RESOLUTION: 512,
    DENSITY_DISSIPATION: 0.97,
    VELOCITY_DISSIPATION: 0.99,
    PRESSURE: 0.8,
    PRESSURE_ITERATIONS: 20,
    CURL: 30,
    SPLAT_RADIUS: 0.25,
    SPLAT_FORCE: 6000,
    SHADING: true,
    COLORFUL: true,
    PAUSED: false,
    BACK_COLOR: { r: 0, g: 0, b: 0 },
    TRANSPARENT: false,
    BLOOM: true,
    BLOOM_THRESHOLD: 0.3,
    BLOOM_STRENGTH: 1.5,
    BLOOM_RADIUS: 0,
  };

  class Fluid {
    constructor(renderer) {
      this.renderer = renderer;
      this.size = new THREE.Vector2(
        renderer.domElement.width,
        renderer.domElement.height
      );
      this.config = config;
      this.pointers = [];
      this.init();
    }

    init() {
      this.createTextures();
      this.createMaterial();
      this.createRenderTarget();
      this.createGeometry();
      this.initGUI();
    }

    createTextures() {
      const { SIM_RESOLUTION, DYE_RESOLUTION } = this.config;
      const type = this.renderer.capabilities.floatGPGPU
        ? THREE.FloatType
        : THREE.HalfFloatType;

      this.dye = createDoubleFBO(
        DYE_RESOLUTION,
        DYE_RESOLUTION,
        type,
        THREE.LinearFilter
      );
      this.velocity = createDoubleFBO(
        SIM_RESOLUTION,
        SIM_RESOLUTION,
        type,
        THREE.LinearFilter
      );
      this.divergence = createFBO(
        SIM_RESOLUTION,
        SIM_RESOLUTION,
        type,
        THREE.NearestFilter,
        THREE.RepeatWrapping
      );
      this.curl = createFBO(
        SIM_RESOLUTION,
        SIM_RESOLUTION,
        type,
        THREE.NearestFilter,
        THREE.RepeatWrapping
      );
      this.pressure = createDoubleFBO(
        SIM_RESOLUTION,
        SIM_RESOLUTION,
        type,
        THREE.NearestFilter,
        THREE.RepeatWrapping
      );

      this.dye.texture.name = "Fluid.dye";
      this.velocity.texture.name = "Fluid.velocity";
      this.divergence.texture.name = "Fluid.divergence";
      this.curl.texture.name = "Fluid.curl";
      this.pressure.texture.name = "Fluid.pressure";
    }

    createMaterial() {
      const { SIM_RESOLUTION, DYE_RESOLUTION } = this.config;
      const type = this.renderer.capabilities.floatGPGPU
        ? THREE.FloatType
        : THREE.HalfFloatType;

      this.material = new THREE.ShaderMaterial({
        uniforms: {
          dyeTexture: { value: null },
          velocityTexture: { value: null },
          time: { value: 0 },
          size: { value: new THREE.Vector2(this.size.x, this.size.y) },
          deltaT: { value: 0 },
          dissipation: { value: config.DENSITY_DISSIPATION },
          velocityDissipation: { value: config.VELOCITY_DISSIPATION },
          pressure: { value: config.PRESSURE },
          curlStrength: { value: config.CURL },
          splatRadius: { value: config.SPLAT_RADIUS },
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
          }
        `,
        fragmentShader: `
          varying vec2 vUv;
          uniform sampler2D dyeTexture;
          uniform sampler2D velocityTexture;
          uniform float time;
          uniform vec2 size;
          uniform float deltaT;
          uniform float dissipation;
          uniform float velocityDissipation;
          uniform float pressure;
          uniform float curlStrength;
          uniform float splatRadius;

          vec2 getUv(vec2 uv) {
            return uv;
          }

          vec4 bilerp(sampler2D sam, vec2 uv, vec2 size) {
            vec2 st = uv / size - 0.5;
            vec2 iuv = floor(st);
            vec2 fuv = fract(st);

            vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * size);
            vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * size);
            vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * size);
            vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * size);

            return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
          }

          void main() {
            vec2 uv = getUv(vUv);
            vec4 dye = texture2D(dyeTexture, uv);
            vec2 velocity = texture2D(velocityTexture, uv).xy;

            // Advection
            vec2 coord = uv - deltaT * velocity * vec2(1.0 / size.x, 1.0 / size.y);
            dye.rgb = bilerp(dyeTexture, coord, vec2(1.0 / ${DYE_RESOLUTION.toFixed(
              1
            )}, 1.0 / ${DYE_RESOLUTION.toFixed(1)})).rgb;

            // Dissipation
            dye.rgb *= dissipation;

            // Velocity dissipation
            velocity *= velocityDissipation;

            // Pressure
            vec2 pressureGradient = vec2(
              (texture2D(pressure.read, uv + vec2(1.0 / size.x, 0.0)).x - texture2D(pressure.read, uv - vec2(1.0 / size.x, 0.0)).x) * 0.5,
              (texture2D(pressure.read, uv + vec2(0.0, 1.0 / size.y)).x - texture2D(pressure.read, uv - vec2(0.0, 1.0 / size.y)).x) * 0.5
            );
            velocity -= deltaT * pressureGradient * pressure;

            // Curl
            float L = texture2D(curl, uv - vec2(1.0 / size.x, 0.0)).x;
            float R = texture2D(curl, uv + vec2(1.0 / size.x, 0.0)).x;
            float T = texture2D(curl, uv + vec2(0.0, 1.0 / size.y)).x;
            float B = texture2D(curl, uv - vec2(0.0, 1.0 / size.y)).x;
            float vorticity = R - L - T + B;
            velocity += deltaT * vec2(T - B, R - L) * curlStrength;

            // Output
            gl_FragColor = vec4(dye.rgb, 1.0);
            gl_FragColor.a = dye.a;

            // Write velocity back to texture
            gl_FragColor.ba = velocity;
          }
        `,
      });
    }

    createRenderTarget() {
      const type = this.renderer.capabilities.floatGPGPU
        ? THREE.FloatType
        : THREE.HalfFloatType;
      this.renderTarget = new THREE.WebGLRenderTarget(
        this.size.x,
        this.size.y,
        {
          wrapS: THREE.ClampToEdgeWrapping,
          wrapT: THREE.ClampToEdgeWrapping,
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          format: THREE.RGBAFormat,
          type: type,
          stencilBuffer: false,
          depthBuffer: false,
        }
      );
    }

    createGeometry() {
      const geometry = new THREE.PlaneGeometry(2, 2);
      this.mesh = new THREE.Mesh(geometry, this.material);
    }

    initGUI() {
      const gui = new GUI();
      gui
        .add(this.config, "DENSITY_DISSIPATION", 0.9, 1, 0.01)
        .name("dye dissipation");
      gui
        .add(this.config, "VELOCITY_DISSIPATION", 0.9, 1, 0.01)
        .name("velocity dissipation");
      gui.add(this.config, "PRESSURE", 0.0, 1, 0.01).name("pressure");
      gui.add(this.config, "CURL", 0, 50, 1).name("curl");
      gui.add(this.config, "SPLAT_RADIUS", 0.01, 1, 0.01).name("splat radius");
      gui.add(this.config, "PAUSED").name("paused");
    }

    update(dt) {
      if (!this.config.PAUSED) {
        this.material.uniforms.time.value += dt;
        this.material.uniforms.deltaT.value = dt;

        this.renderer.setRenderTarget(this.dye.write);
        this.renderer.render(this.mesh, this.camera);
        this.dye.swap();

        this.renderer.setRenderTarget(this.velocity.write);
        this.renderer.render(this.mesh, this.camera);
        this.velocity.swap();

        this.renderer.setRenderTarget(this.divergence);
        this.renderer.render(this.mesh, this.camera);

        this.renderer.setRenderTarget(this.curl);
        this.renderer.render(this.mesh, this.camera);

        this.renderer.setRenderTarget(this.pressure.write);
        this.renderer.render(this.mesh, this.camera);
        this.pressure.swap();

        this.renderer.setRenderTarget(null);
      }
    }

    splat(x, y, dx, dy, color) {
      const { SIM_RESOLUTION, DYE_RESOLUTION, SPLAT_FORCE } = this.config;
      const uniforms = this.material.uniforms;

      uniforms.dyeTexture.value = this.dye.read.texture;
      uniforms.velocityTexture.value = this.velocity.read.texture;

      uniforms.splatRadius.value = config.SPLAT_RADIUS / 100;
      uniforms.splatForce.value =
        SPLAT_FORCE * Math.max(this.size.x, this.size.y);
      this.renderer.setRenderTarget(this.dye.write);
      this.renderer.render(this.mesh, this.camera);
      this.dye.swap();

      uniforms.splatRadius.value = config.SPLAT_RADIUS / 100;
      uniforms.splatForce.value =
        SPLAT_FORCE * Math.max(this.size.x, this.size.y);
      this.renderer.setRenderTarget(this.velocity.write);
      this.renderer.render(this.mesh, this.camera);
      this.velocity.swap();

      this.renderer.setRenderTarget(null);
    }
  }

  function createFBO(w, h, type, minFilter, wrap) {
    const texture = new THREE.DataTexture(
      new Float32Array(4 * w * h),
      w,
      h,
      THREE.RGBAFormat,
      type,
      undefined,
      wrap,
      wrap,
      minFilter,
      THREE.LinearFilter
    );
    texture.needsUpdate = true;

    const target = new THREE.WebGLRenderTarget(w, h, {
      wrapS: wrap,
      wrapT: wrap,
      minFilter: minFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: type,
      stencilBuffer: false,
      depthBuffer: false,
    });

    return {
      texture,
      target,
    };
  }

  function createDoubleFBO(w, h, type, minFilter) {
    let fbo1 = createFBO(w, h, type, minFilter);
    let fbo2 = createFBO(w, h, type, minFilter);

    return {
      read: fbo1,
      write: fbo2,
      swap: () => {
        const temp = fbo1;
        fbo1 = fbo2;
        fbo2 = temp;
      },
    };
  }

  // --- End Fluid Simulation ---

  useEffect(() => {
    // Initialize THREE.js scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Initialize camera
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    cameraRef.current = camera;

    // Initialize renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      canvas: canvasRef.current,
      preserveDrawingBuffer: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    // Initialize effect composer
    const composer = new EffectComposer(renderer);
    composerRef.current = composer;

    // Render pass
    composer.addPass(new RenderPass(scene, camera));

    // Fluid simulation
    fluidSimRef.current = new Fluid(renderer);

    // Background material
    const textureLoader = new THREE.TextureLoader();
    const backgroundTexture = textureLoader.load("/images/bg4.png");
    const logoTexture = textureLoader.load("/images/logo.png");

    const material = new THREE.ShaderMaterial({
      extensions: {
        derivatives: true,
      },
      uniforms: {
        tBg: { value: backgroundTexture },
        tLogo: { value: logoTexture },
        uColorBg: { value: new THREE.Color("#000") },
        uColorLogo: { value: new THREE.Color("#fff") },
        uNoise: { value: 0 },
        uNoise1Opts: { value: new THREE.Vector2(1.25, 0.25) },
        uNoise2Opts: { value: new THREE.Vector2(2, 0.8) },
        uNoise3Opts: { value: new THREE.Vector3(5, 2, 3.8) },
        uNoise4Opts: { value: new THREE.Vector4(-3.8, -2, -3.9, -2.5) },
        uGlobalShape: { value: 0 },
        uGlobalOpen: { value: 0 },
        uNoiseMultiplier: { value: 0 },
        uDye: fluidSimRef.current.dye.read.texture,
        uVel: fluidSimRef.current.velocity.read.texture,
        uUV: { value: null },
        uLogoAnimation: { value: 0 },
        resolution: {
          value: new THREE.Vector2(window.innerWidth, window.innerHeight),
        },
        time: { value: 0 },
      },
      vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position, 1.0);
          }
        `,
      fragmentShader: `
          varying vec2 vUv;
          uniform sampler2D tBg;
          uniform sampler2D tLogo;
          uniform vec3 uColorBg;
          uniform vec3 uColorLogo;
          uniform float uNoise;
          uniform vec2 uNoise1Opts;
          uniform vec2 uNoise2Opts;
          uniform vec3 uNoise3Opts;
          uniform vec4 uNoise4Opts;
          uniform float uGlobalShape;
          uniform float uGlobalOpen;
          uniform float uNoiseMultiplier;
          uniform sampler2D uDye;
          uniform sampler2D uVel;
          uniform float uLogoAnimation;
          uniform vec2 resolution;
          uniform float time;

          // Place fragment shader code here...
        `,
      transparent: false,
    });
    materialRef.current = material;
    mainUniformsRef.current = material.uniforms;

    // Plane geometry
    const geometry = new THREE.PlaneGeometry(2, 2);

    // Create mesh and add to scene
    const mesh = new THREE.Mesh(geometry, material);
    meshRef.current = mesh;
    scene.add(mesh);

    // Elastic effect composer
    const elasticRenderTarget = new THREE.WebGLRenderTarget(2, 2, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: renderer.capabilities.floatGPGPU
        ? THREE.FloatType
        : THREE.HalfFloatType,
    });
    elasticComposerRef.current = new EffectComposer(
      renderer,
      elasticRenderTarget
    );
    elasticComposerRef.current.renderToScreen = false;

    const elasticPass = new ShaderPass({
      uniforms: {
        tDiffuse: { value: null },
        tVel: fluidSimRef.current.velocity.read.texture,
        resolution: {
          value: new THREE.Vector2(window.innerWidth, window.innerHeight),
        },
        time: { value: 0 },
        dtRatio: { value: 1 },
      },
      vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position, 1.0);
          }
        `,
      fragmentShader: `
          varying vec2 vUv;
          uniform sampler2D tDiffuse;
          uniform sampler2D tVel;
          uniform vec2 resolution;
          uniform float time;
          uniform float dtRatio;

          // Place fragment shader code here...
        `,
      depthWrite: true,
      depthTest: false,
    });
    elasticComposerRef.current.addPass(elasticPass);

    // Event listeners
    window.addEventListener("resize", onWindowResize);
    document.addEventListener("pointermove", onPointerMove);

    // Animation loop
    animate();

    return () => {
      // Cleanup
      window.removeEventListener("resize", onWindowResize);
      document.removeEventListener("pointermove", onPointerMove);
      cancelAnimationFrame(requestAnimationFrameId);
      renderer.dispose();
      composer.dispose();
      elasticComposerRef.current.dispose();
      fluidSimRef.current.dispose();
    };
  }, []);

  // --- Event Handlers ---
  const onWindowResize = () => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();

    rendererRef.current.setSize(width, height);
    composerRef.current.setSize(width, height);
    elasticComposerRef.current.setSize(width, height);

    fluidSimRef.current.size.x = width;
    fluidSimRef.current.size.y = height;
    fluidSimRef.current.createRenderTarget();

    materialRef.current.uniforms.resolution.value.set(width, height);
  };

  const onPointerMove = (e) => {
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    fluidSimRef.current.splat(
      x,
      y,
      e.movementX,
      e.movementY,
      new THREE.Color(1, 1, 1)
    );
  };

  // --- Animation Loop ---
  let lastTime = performance.now();
  let requestAnimationFrameId;

  const animate = () => {
    requestAnimationFrameId = requestAnimationFrame(animate);

    const time = performance.now();
    const dt = (time - lastTime) / 1000;
    lastTime = time;

    // Update fluid simulation
    fluidSimRef.current.update(dt);

    // Update elastic effect
    elasticComposerRef.current.render(dt);

    // Update background material uniforms
    mainUniformsRef.current.uUV.value =
      elasticComposerRef.current.readBuffer.texture;
    mainUniformsRef.current.time.value = time;

    // Render scene
    composerRef.current.render();
  };

  // --- Render ---
  return (
    <div ref={containerRef} className="fluid-background">
      <canvas ref={canvasRef} />
    </div>
  );
};

export default FluidBackground;
