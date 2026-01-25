/**
 * NAVODILA
 * - samo odpreš index.html v browserju po tem ko zalufaš backend server
 * */

import * as THREE from './three.js';
import {OrbitControls} from './OrbitControls.js';
import {OBJLoader} from './OBJLoader.js';
import {MTLLoader} from './MTLLoader.js';
import {Sky} from './Sky.js';

let renderer, scene, camera;
let car = null;
let ground = null;
let carSpeed = 0.03;

// templatei za kloniranje - dodaj nove objekte kot template in v otherObjects ko jih izrisuješ
let carTemplate = null;
let parkingTemplate = null;
let wallTemplate = null;
let humanTemplate = null;

// luči
let sunLight = null;
let ambientLight = null;
let headlightGroup = null;
let hemiLight = null;
let envRT = null;
let envCam = null;
let moon = null;
let isDarkMode = false;

/* izbljšave 15.1 naprej*/

// texture loader
const texLoader = new THREE.TextureLoader();

// moon
let moonColor, moonNormal, moonRough, moonMesh;
// sky
let sky;
let stars;
const pools = {
    car: [],
    human: [],
    parking: [],
    wall: []
};

let groundOffsetZ = 0;
let displayYoloResult = false;

let parkingOverlayTextures;
let lastTime = performance.now();
let frameCount = 0;
let fps = 0;
let quality = "high"; // "high" | "medium" | "low"
const modelLoadTimes = {};
let lastFrameTime = performance.now();
let frameTimes = [];

const FRONT_X = 12;
const BACK_X = 24;
const MIN = 0;   // observed min left_to_right
const MAX = 100;   // observed max left_to_right
const INTERPOLATION_AMOUNT = 13;

let canParkInvalid = false;
let canParkDruzina = false;

init();
animate();

function init() {
    renderer = new THREE.WebGLRenderer();

    // shadows enable
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;

    // set size and add to document
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // scena
    scene = new THREE.Scene();

    // POGLED KAMERE
    camera = new THREE.PerspectiveCamera(
        75, // FOV
        window.innerWidth / window.innerHeight, // aspect
        0.1, // blizu
        1000 // daleč
    );
    camera.position.set(-25, 15, 20);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    const hemi = new THREE.HemisphereLight(
        0x3a5fa0,  // sky blue
        0x0a0a0a,  // ground
        0.45       // stronger bounce
    );
    scene.add(hemi);

    // TLA
    const asphaltColor = texLoader.load('textures/asphalt/asphalt_04_diff_4k.jpg');
    const asphaltNormal = texLoader.load('textures/asphalt/asphalt_04_nor_gl_4k.exr');
    const asphaltRough = texLoader.load('textures/asphalt/asphalt_04_rough_4k.exr');

    parkingOverlayTextures = {
        electric: texLoader.load('textures/parking_slike/electric.png'),
        invalid: texLoader.load('textures/parking_slike/invalid.jpg'),
        druzina: texLoader.load('textures/parking_slike/druzina.png')
    };
    Object.values(parkingOverlayTextures).forEach(t => {
        t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
        t.anisotropy = 4;
    });

    [asphaltColor, asphaltNormal, asphaltRough].forEach(t => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.repeat.set(1, 20); // long road tiling
    });

    const groundMat = new THREE.MeshStandardMaterial({
        map: asphaltColor,
        normalMap: asphaltNormal,
        roughnessMap: asphaltRough,
        roughness: 1.0,
        metalness: 0.0
    });

    ground = new THREE.Mesh(
        new THREE.BoxGeometry(100, 1, 1000),
        groundMat
    );

    ground.castShadow = false;
    ground.receiveShadow = true;
    ground.position.y = -2;

    ground.material.envMap = null;
    ground.material.envMapIntensity = 0;
    ground.material.needsUpdate = true;
    ground.material.normalScale.set(0.4, 0.4);

    scene.add(ground);

    sky = new Sky();
    sky.scale.setScalar(10000);
    scene.add(sky);
    createStars();

    moonColor = texLoader.load('textures/moon/moon_01_diff_4k.jpg');
    moonNormal = texLoader.load('textures/moon/moon_01_nor_gl_4k.exr');
    moonRough = texLoader.load('textures/moon/moon_01_rough_4k.exr');

    // pred-naložimo avto
    preloadModel("objects-models/rac_grafika_model_armatura2.mtl", "objects-models/rac_grafika_model_armatura2.obj", (obj) => {
        carTemplate = obj;
        carTemplate.rotation.y += Math.PI / 2;

        // glavni avto
        car = cloneObject(carTemplate);
        car.position.set(0, 0, 0);
        scene.add(car);

        addHeadlightsToCar(car);
    });

    // pred-naložimo parkirno mesto
    // TODO rabimo dodati invalidska
    preloadModel("objects-models/x.mtl", "objects-models/parkirna_mesta.obj", (obj) => {
        parkingTemplate = obj;
        parkingTemplate.rotation.y -= Math.PI / 2;
    });

    // pred-naložimo stebre in ostale objekte
    preloadModel("objects-models/human.mtl", "objects-models/human.obj", (obj) => {
        humanTemplate = obj;
    });
    preloadModel("objects-models/x.mtl", "objects-models/steber.obj", (obj) => {
        wallTemplate = obj;
    });

    lightMode();

    if (envCam) {
        envCam.update(renderer, scene);
    }

    window.addEventListener('resize', onWindowResize);

    setTimeout(() => {
        console.table(modelLoadTimes);
    }, 3000);
}

function preloadModel(mtl, objPath, callback) {
    const start = performance.now();
    const mtlLoader = new MTLLoader();

    mtlLoader.load(mtl, (materials) => {
        materials.preload();
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);

        objLoader.load(objPath, (object) => {
            const box = new THREE.Box3().setFromObject(object);
            const center = new THREE.Vector3();
            box.getCenter(center);

            object.position.sub(center);

            const wrapper = new THREE.Group();
            wrapper.add(object);

            wrapper.traverse(child => {
                if (child instanceof THREE.Mesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            const end = performance.now();
            console.log(`[modelload time] ${objPath}: ${Math.round(end - start)} ms`);

            callback(wrapper);
        });
    });
}

function cloneObject(template) {
    const clone = template.clone(true);

    clone.traverse((child) => {
        if (!child.isMesh) return;

        if (child.material && child.material.clone) {
            child.material = child.material.clone();
            return;
        }

        if (Array.isArray(child.material)) {
            child.material = child.material.map(m =>
                m && m.clone ? m.clone() : m
            );
            return;
        }

        // fallback ko ni materialov
        if (!child.material) {
            child.material = new THREE.MeshLambertMaterial({color: 0xffffff});
            return;
        }

        child.material = new THREE.MeshLambertMaterial({
            color: child.material.color || 0xffffff
        });

        child.castShadow = true;
        child.receiveShadow = true;
    });

    scene.add(clone);
    return clone;
}

// websocke povezav
const ws = new WebSocket("ws://localhost:8000/ws/frontend");

ws.onmessage = (msg) => {
    let stats = {
        cars: 0,
        humans: 0,
        parkings: 0
    };
    let canParkInCurrentFrame = false;

    const data = JSON.parse(msg.data);
    const detections = data.detections;

    if (displayYoloResult && data.image) {
        const img = document.getElementById("camera-feed");
        img.src = 'data:image/jpeg;base64,' + data.image;
    }

    if (!detections) return;

    const used = {
        car: new Set(),
        human: new Set(),
        parking: new Set(),
        wall: new Set()
    };

    detections.forEach(det => {
        if (det.label === "Avtomobil" && carTemplate) {
            const t = THREE.MathUtils.clamp(
                (det.left_to_right - MIN) / (MAX - MIN),
                0,
                1
            );

            const zPos = THREE.MathUtils.lerp(-INTERPOLATION_AMOUNT, INTERPOLATION_AMOUNT, t);
            let xPos;

            stats.cars++;
            if (det.depth_row === "front") {
                xPos = FRONT_X;
            } else {
                xPos = BACK_X;
            }

            const car = getFromPool("car", carTemplate);
            car.position.set(xPos, 0, zPos);

            if (isDarkMode && !car.userData.glowAdded) {
                addCarGlow(car);
                car.userData.glowAdded = true;
            }
            used.car.add(car);
        }

        if (det.label.toLowerCase().includes("lovek") && humanTemplate) {
            const t = THREE.MathUtils.clamp(
                (det.left_to_right - MIN) / (MAX - MIN),
                0,
                1
            );

            const zPos = THREE.MathUtils.lerp(-INTERPOLATION_AMOUNT, INTERPOLATION_AMOUNT, t);
            let xPos;
            stats.humans++;

            const human = getFromPool("human", humanTemplate);
            if (det.depth_row === "front") {
                xPos = FRONT_X;
            } else {
                xPos = BACK_X;
            }

            human.position.set(xPos, 0, zPos);
            used.human.add(human);
        }

        if (det.label.toLowerCase().includes("steber") && wallTemplate) {
            const t = THREE.MathUtils.clamp(
                (det.left_to_right - MIN) / (MAX - MIN),
                0,
                1
            );

            const zPos = THREE.MathUtils.lerp(-INTERPOLATION_AMOUNT, INTERPOLATION_AMOUNT, t);
            let xPos;
            console.log('steber zPos', zPos)

            const wall = getFromPool("wall", wallTemplate);
            if (det.depth_row === "front") {
                xPos = FRONT_X;
            } else {
                xPos = BACK_X;
            }

            wall.position.set(xPos, 0, zPos);
            used.wall.add(wall);
        }


        if (det.label.toLowerCase().includes("parki") && parkingTemplate) {
            const t = THREE.MathUtils.clamp(
                (det.left_to_right - MIN) / (MAX - MIN),
                0,
                1
            );

            const zPos = THREE.MathUtils.lerp(-INTERPOLATION_AMOUNT, INTERPOLATION_AMOUNT, t);
            console.log('parking zPos', zPos)
            let xPos;

            stats.parkings ++;
            const p = getFromPool("parking", parkingTemplate);

            xPos = FRONT_X;
            p.position.set(xPos, 0, zPos);

            const label = det.label.toLowerCase();

            if (label.includes("ele")) {
                addParkingOverlay(p, "electric");
            } else if (label.includes("dru")) {
                addParkingOverlay(p, "druzina");
            } else if (label.includes("valid")) {
                addParkingOverlay(p, "invalid");
            } else {
                removeParkingOverlay(p);
            }

            if (isDarkMode && !p.userData.glowAdded) {
                addParkingGlow(p);
                p.userData.glowAdded = true;
            }
            used.parking.add(p);

            let isSpotLegal = true;

            if (label.includes("valid") && !canParkInvalid) {
                isSpotLegal = false;
            } else if (label.includes("dru") && !canParkDruzina) {
                isSpotLegal = false;
            }

            if (isSpotLegal) {
                canParkInCurrentFrame = true;
            }
        }
    });

    const border = document.getElementById("status-border");
    if (stats.parkings === 0) {
        border.style.borderColor = "transparent";
    } else if (canParkInCurrentFrame) {
        border.style.borderColor = "#00ff66";
    } else {
        border.style.borderColor = "#ff0000";
    }

    releaseUnused(pools.car, used.car);
    releaseUnused(pools.human, used.human);
    releaseUnused(pools.wall, used.wall);
    releaseUnused(pools.parking, used.parking);
    updateHUD(stats);
};

function animate() {
    requestAnimationFrame(animate);

    // FPS calculation
    const now = performance.now();
    const frameTime = now - lastFrameTime;
    lastFrameTime = now;
    frameCount++;

    frameTimes.push(frameTime);

    // keep last 300 frames (~5s)
    if (frameTimes.length > 300) {
        frameTimes.shift();
    }


    if (now - lastTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastTime = now;
        document.getElementById("hud-fps").innerText = fps;
    }

    if (car) {
        groundOffsetZ += carSpeed;
        const ROAD_LENGTH = 1000;
        ground.position.z = groundOffsetZ % ROAD_LENGTH;
    }

    renderer.render(scene, camera);
}

function updateHUD(stats) {
    document.getElementById("hud-mode").innerText =
        isDarkMode ? "Night" : "Day";

    document.getElementById("hud-cars").innerText =
        stats.cars;

    document.getElementById("hud-humans").innerText =
        stats.humans;

        document.getElementById("hud-parkings").innerText =
        stats.parkings;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function addHeadlightsToCar(car) {
    if (headlightGroup) car.remove(headlightGroup);

    headlightGroup = new THREE.Group();

    const mk = () => new THREE.SpotLight(
        0xffffff,
        250,        // intensity
        200,          // distance: 0 = infinite
        1.5,  // angle: wide cone
        1.6,        // penumbra
        1           // decay: less falloff
    );

    const leftLight = mk();
    const rightLight = mk();

    leftLight.position.set(2, 1.0, -8);
    rightLight.position.set(2, 1.0, -4);

    const leftTarget = new THREE.Object3D();
    const rightTarget = new THREE.Object3D();

    car.add(leftTarget);
    car.add(rightTarget);

    leftTarget.position.set(20, 0.2, -13.9);
    rightTarget.position.set(100, 0.2, -13.9);

    leftLight.target = leftTarget;
    rightLight.target = rightTarget;

    headlightGroup.add(leftLight);
    headlightGroup.add(rightLight);

    const bulbGeo = new THREE.SphereGeometry(0.08, 12, 12);
    const bulbMat = new THREE.MeshBasicMaterial({color: 0xffffff});
    const bulbL = new THREE.Mesh(bulbGeo, bulbMat);
    const bulbR = new THREE.Mesh(bulbGeo, bulbMat);
    bulbL.position.copy(leftLight.position);
    bulbR.position.copy(rightLight.position);
    headlightGroup.add(bulbL, bulbR);

    car.add(headlightGroup);

    headlightGroup.visible = false;

    // DEBUGGIRANJE
    /*const leftHelper = new THREE.SpotLightHelper(leftLight);
    const rightHelper = new THREE.SpotLightHelper(rightLight);
    scene.add(leftHelper, rightHelper);
    leftLight.userData.helper = leftHelper;
    rightLight.userData.helper = rightHelper;*/
}

function darkMode() {
    scene.background = new THREE.Color(0x02050c); // skoraj črno nebo

    // ambientna luč
    ambientLight = new THREE.AmbientLight(0x0b10ff, 0.52);
    scene.add(ambientLight);

    // "luna" – zelo šibka globalna svetloba
    sunLight = new THREE.DirectionalLight(0x9fb4ff, 0.48);
    sunLight.position.set(-80, 120, -40);
    sunLight.castShadow = true;
    // resolucija sence
    sunLight.shadow.mapSize.set(1024, 1024);
    // fixes shadow acne
    sunLight.shadow.bias = -0.0004;
    scene.add(sunLight);

    // barva neeba
    hemiLight = new THREE.HemisphereLight(
        0x0b1d3a,
        0x000000,
        0.08
    );
    scene.add(hemiLight);

    // močna nočna megla
    scene.fog = new THREE.FogExp2(0x02050c, 0.016);

    // environment temen
    if (envCam) {
        scene.remove(envCam);
        envCam = null;
    }
    envRT = new THREE.WebGLCubeRenderTarget(64);
    envCam = new THREE.CubeCamera(0.1, 500, envRT);
    scene.add(envCam);
    scene.environment = envRT.texture;

    // tone mapping – temna scena
    // ACES -> realisičen contrast in smooth highlights
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.7;

    // luna kot vizualni element

    if (!moonMesh) {
    const moonMat = new THREE.MeshStandardMaterial({
        map: moonColor,
        normalMap: moonNormal,
        roughnessMap: moonRough,
        roughness: 0.5,
        metalness: 0.5,
        color: 0xfff1c1
    });

    moonMesh = new THREE.Mesh(
        new THREE.SphereGeometry(5, 32, 32),
        moonMat
    );
        moonMesh.position.set(-70, 80, -100);
        scene.add(moonMesh);
    }
    moonMesh.visible = true;

    if (headlightGroup) {
        headlightGroup.visible = true;
    }

    sky.visible = true;

    sky.material.uniforms.turbidity.value = 1;
    sky.material.uniforms.rayleigh.value = 0.2;
    sky.material.uniforms.mieCoefficient.value = 0.001;
    sky.material.uniforms.mieDirectionalG.value = 0.7;

    // moon position (night)
    const moonDir = new THREE.Vector3().setFromSphericalCoords(
        1,
        Math.PI * 0.85,
        Math.PI * 0.5
    );
    sky.material.uniforms.sunPosition.value.copy(moonDir);
    renderer.shadowMap.enabled = false;

    stars.visible = true;
}

function addParkingGlow(space) {
    const light = new THREE.PointLight(
        0x00ff66, // vivid green
        3,
        40,
        1
    );
    light.position.set(0, 1.5, 0);
    space.add(light);

    // visible glow source - nanj ne vpliva osvetlitev scene
    const glowGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0x00ff66
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.copy(light.position);
    space.add(glow);

    // boost emissive on parking lines
    space.traverse(child => {
        if (!child.isMesh) return;
        if (!child.material) return;

        if (child.material.emissive) {
            child.material.emissive.set(0x00ff66);
            child.material.emissiveIntensity = 1.2;
        }
    });

    space.userData.glowLight = light;
}

function addCarGlow(space) {
    const light = new THREE.PointLight(
        0xff0000,
        1.2,
        12,
        1
    );
    light.position.set(0, 2.5, 0);
    space.add(light);

    // visible glow source
    const glowGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0xff0000
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.copy(light.position);
    space.add(glow);

    // boost emissive on parking lines
    space.traverse(child => {
        if (!child.isMesh) return;
        if (!child.material) return;

        if (child.material.emissive) {
            child.material.emissive.set(0xff0000);
            child.material.emissiveIntensity = 1.2;
        }
    });

    space.userData.glowLight = light;
}

function lightMode() {
    scene.background = new THREE.Color(0x87bfff); // močno dnevno nebo

    // odstranimo meglo
    scene.fog = null;

    // ambient samo kot fill
    ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    scene.add(ambientLight);

    // močno sonce
    sunLight = new THREE.DirectionalLight(0xfff2d6, 1.4);
    sunLight.position.set(-120, 150, 60);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(2048, 2048);
    sunLight.shadow.bias = -0.0003;
    scene.add(sunLight);

    // zelo šibek hemisphere bounce
    hemiLight = new THREE.HemisphereLight(
        0xcfe7ff, // nebo
        0xffffff, // tla
        0.15
    );
    scene.add(hemiLight);

    // okoljski odboji
    // bolj podrobni
    if (envCam) {
        scene.remove(envCam);
        envCam = null;
    }
    envRT = new THREE.WebGLCubeRenderTarget(128);
    envCam = new THREE.CubeCamera(0.1, 1000, envRT);
    scene.add(envCam);
    scene.environment = envRT.texture;

    // tone mapping – realen dan
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;

    // odstrani luno če obstaja
    if (moon) {
        if (moonMesh) moonMesh.visible = false;
        scene.remove(moon);
        moon = null;
    }
    scene.fog = null;

    if (headlightGroup) {
        headlightGroup.visible = false;
    }

    sky.visible = true;

    sky.material.uniforms.turbidity.value = 8;
    sky.material.uniforms.rayleigh.value = 2;
    sky.material.uniforms.mieCoefficient.value = 0.005;
    sky.material.uniforms.mieDirectionalG.value = 0.8;

    // sun position (day)
    const sun = new THREE.Vector3().setFromSphericalCoords(
        1,
        Math.PI * 0.45,
        Math.PI * 0.25
    );
    sky.material.uniforms.sunPosition.value.copy(sun);
    stars.visible = false;

    ["car", "parking"].forEach(type => {
        pools[type].forEach(obj => {
            if (!obj.visible) return;
            removeParkingGlow(obj);
            obj.userData.glowAdded = false;
        });
    });

    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
}

function toggleMode() {
    clearLighting();

    if (isDarkMode) {
        lightMode();
    } else {
        darkMode();
    }

    isDarkMode = !isDarkMode;

    if (envCam) {
        envCam.update(renderer, scene);
    }
}

function clearLighting() {
    if (ambientLight) scene.remove(ambientLight);
    if (sunLight) scene.remove(sunLight);
    if (hemiLight) scene.remove(hemiLight);
    if (moon) scene.remove(moon);

    ambientLight = null;
    sunLight = null;
    hemiLight = null;
    moon = null;

    scene.fog = null;
}

window.addEventListener('keydown', (e) => {
    if (e.key === 'o' || e.key === 'O') {
        toggleMode();
    }

    if (e.key === 'y' || e.key === 'Y') {
        displayYoloResult = !displayYoloResult;
        if (!displayYoloResult) {
            const img = document.getElementById("image");
            img.style.display = 'none';
        } else {
            const img = document.getElementById("image");
            img.style.display = 'block';
        }
    }

    if (e.key === '1') applyQuality("low");
    if (e.key === '2') applyQuality("medium");
    if (e.key === '3') applyQuality("high");

});

function getFromPool(type, template) {
    const pool = pools[type];

    let obj = pool.find(o => !o.visible);
    if (!obj) {
        obj = cloneObject(template);

        if (type === "car") {
            obj.rotation.y -= Math.PI / 2;
            obj.userData.rotated = true;
        }

        pool.push(obj);
    }

    obj.visible = true;
    return obj;
}

function releaseUnused(pool, usedSet) {
    pool.forEach(o => {
        if (!usedSet.has(o)) {
            o.visible = false;

            removeParkingOverlay(o);
            removeParkingGlow(o);

            o.userData.glowAdded = false;
        }
    });
}


function createStars() {
    const count = 500;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const r = 200;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(THREE.MathUtils.randFloat(-0.2, 1)); // fewer near horizon

        pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        pos[i * 3 + 1] = r * Math.cos(phi);
        pos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }

    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

    const mat = new THREE.PointsMaterial({
        color: 0xffff00 ,        // warm yellow
        size: 2.0,              // IMPORTANT (0.7 is too small)
        transparent: true,
        opacity: 1.0,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    stars = new THREE.Points(geo, mat);
    stars.visible = false;
    scene.add(stars);
}

function addParkingOverlay(parkingObj, type) {
    if (!parkingOverlayTextures[type]) return;

    removeParkingOverlay(parkingObj);

    let baseMesh = null;
    parkingObj.traverse(c => {
        if (c.isMesh && !baseMesh) baseMesh = c;
    });
    if (!baseMesh) return;

    const box = new THREE.Box3().setFromObject(baseMesh);
    const size = new THREE.Vector3();
    box.getSize(size);

    const geo = new THREE.PlaneGeometry(
        size.x * 0.5,
        size.z * 0.7
    );

    const mat = new THREE.MeshBasicMaterial({
        map: parkingOverlayTextures[type],
        transparent: true,
        depthWrite: false
    });

    const overlay = new THREE.Mesh(geo, mat);

    overlay.rotation.x = -Math.PI / 2;
    overlay.position.set(
        0,
        box.max.y - baseMesh.position.y + 0.01,
        0
    );

    overlay.renderOrder = 10;

    parkingObj.add(overlay);
    parkingObj.userData.overlay = overlay;
    parkingObj.userData.overlayType = type;
}

function removeParkingOverlay(parkingObj) {
    if (!parkingObj.userData.overlay) return;

    parkingObj.remove(parkingObj.userData.overlay);
    parkingObj.userData.overlay = null;
    parkingObj.userData.overlayType = null;
}

function removeParkingGlow(space) {
    if (!space.userData.glowLight) return;

    space.remove(space.userData.glowLight);
    space.userData.glowLight = null;

    space.children = space.children.filter(c => {
        if (c.isMesh && c.material?.color?.getHex() === 0x00ff66) {
            c.geometry.dispose();
            c.material.dispose();
            return false;
        }
        return true;
    });
}

function applyQuality(level) {
    quality = level;

    if (level === "high") {
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFShadowMap;
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    if (level === "medium") {
        renderer.setPixelRatio(1);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.BasicShadowMap;
        renderer.setSize(window.innerWidth * 0.85, window.innerHeight * 0.85, false);
    }

    if (level === "low") {
        renderer.setPixelRatio(0.75);
        renderer.shadowMap.enabled = false;
        renderer.setSize(window.innerWidth * 0.7, window.innerHeight * 0.7, false);
    }

    document.getElementById("hud-quality").innerText = quality;
}

setInterval(() => {
    if (!frameTimes.length) return;

    const avg =
        frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;

    console.log("Avg frame time:", avg.toFixed(2), "ms");
    document.getElementById("frame-time").innerText = avg.toFixed(2);

}, 2000);


const goBtn = document.getElementById("goBtn");

goBtn.addEventListener("click", () => {
    window.location.href = "http://localhost:8000/vzvratni_public/"; 
});

// UI Logic for Dropdown
document.getElementById('settings-icon').onclick = () => {
    const dropdown = document.getElementById('settings-dropdown');
    dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
};

document.getElementById('allow-invalid').onchange = (e) => {
    canParkInvalid = e.target.checked;
};

document.getElementById('allow-druzina').onchange = (e) => {
    canParkDruzina = e.target.checked;
};