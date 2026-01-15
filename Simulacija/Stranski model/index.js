/**
 * NAVODILA
 * - samo odpreš index.html v browserju po tem ko zalufaš backend server
 * */

import * as THREE from './three.js';
import { OrbitControls } from './OrbitControls.js';
import { OBJLoader } from './OBJLoader.js';
import { MTLLoader } from './MTLLoader.js';

let renderer, scene, camera;
let car = null;
let ground = null;
let carSpeed = 0.005;

// templatei za kloniranje - dodaj nove objekte kot template in v otherObjects ko jih izrisuješ
let carTemplate = null;
let parkingTemplate = null;
let wallTemplate = null;
let humanTemplate = null;

let otherObjects = [];
let parkingSpaces = [];
let parkedCars = [];

// luči
let sunLight = null;
let ambientLight = null;
let headlightGroup = null;
let hemiLight = null;
let envRT = null;
let envCam = null;
let moon = null;
let isDarkMode = false;

// aimacija obračanja kol
const WHEEL_RADIUS = 0.6;
let carWheels = [];

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
    camera.position.set(-30, 20, 40);
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
    ground = new THREE.Mesh(
        new THREE.BoxGeometry(100, 1, 1000),
        new THREE.MeshLambertMaterial({ color: 0x444444 })
    );
    ground.castShadow = true;
    ground.receiveShadow = true;
    ground.position.y = -1.2;
    scene.add(ground);

    // pred-naložimo avto
    preloadModel("objects-models/rac_grafika_model_armatura2.mtl", "objects-models/rac_grafika_model_armatura2.obj", (obj) => {
        carTemplate = obj;
        carTemplate.rotation.y += Math.PI / 2;

        // glavni avto
        car = cloneObject(carTemplate);
        car.position.set(0, 0, 0);
        scene.add(car);
        carWheels = getCarWheels(car);

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

    darkMode();

    if (envCam) {
        envCam.update(renderer, scene);
    }

    window.addEventListener('resize', onWindowResize);
}

function preloadModel(mtl, obj, callback) {
    const mtlLoader = new MTLLoader();
    mtlLoader.load(mtl, (materials) => {
        materials.preload();
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load(obj, (object) => {
            object.traverse(child => {
                if (child instanceof THREE.Mesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            callback(object);
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
            child.material = new THREE.MeshLambertMaterial({ color: 0xffffff });
            return;
        }

        child.material = new THREE.MeshLambertMaterial({
            color: child.material.color || 0xffffff
        });
    });

    scene.add(clone);
    return clone;
}

// TODO mogoče namesto risanja + brisanja samo cachiramo in premikamo objekte
function removeObject(obj) {
    if (!obj) return;

    scene.remove(obj);

    obj.traverse(child => {
        if (!child.isMesh) return;

        if (child.geometry) child.geometry.dispose();

        if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose && m.dispose());
        } else if (child.material && child.material.dispose) {
            child.material.dispose();
        }
    });
}

// TODO isto kot eno gor
function clearSpawnedObjects() {
    parkedCars.forEach(obj => removeObject(obj));
    parkedCars.length = 0;

    parkingSpaces.forEach(obj => removeObject(obj));
    parkingSpaces.length = 0;

    otherObjects.forEach(obj => removeObject(obj));
    otherObjects.length = 0;
}


// websocke povezav
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    const detections = data.detections;

    if (!detections || !detections.length) return;

    clearSpawnedObjects();

    // TODO filtriramo avtomobile v ozadju in ostale detekcije -> mogoče kr na bcakendu
    detections.forEach(det => {
        const zPos = THREE.MathUtils.lerp(-20, 20, det.left_to_right / 100);
        let xPos = THREE.MathUtils.lerp(10, 30, det.down_to_up / 100);

        if (det.label === "Avtomobil" && carTemplate) {
            // malo površno ampak zaenkrat ok
            if (xPos > 22) {
                xPos = 30;
            }

            const newCar = cloneObject(carTemplate);

            if (isDarkMode) {
                addCarGlow(newCar);
            }

            newCar.rotation.y -= Math.PI / 2;
            newCar.position.set(xPos, 0, zPos);
            parkedCars.push(newCar);
        }

        if (det.label.toLowerCase().includes("steber") && wallTemplate) {
            let xPos = THREE.MathUtils.lerp(10, 30, det.coordinates / 100);

            // če je sredina objekt v zgornji četrtini potem je v odzadju drugače spredaj
            if (xPos > 25) {
                xPos = 10;
            } else {
                xPos = 30;
            }
            const newWall = cloneObject(wallTemplate);

            newWall.position.set(xPos, 0, zPos);
            otherObjects.push(newWall);
        }

        if (det.label.toLowerCase().includes("lovek") && humanTemplate) {
            const newHuman = cloneObject(humanTemplate);

            newHuman.position.set(xPos, 0, zPos);
            otherObjects.push(newHuman);
        }

        // TODO bolj precizno + različni parkingi
        if (det.label.toLowerCase().includes("parki") && parkingTemplate) {
            const newParking = cloneObject(parkingTemplate);

            newParking.position.set(10, 0, zPos);

            if (isDarkMode) {
                addParkingGlow(newParking);
            }

            parkingSpaces.push(newParking);
        }
    });
};

function animate() {
    requestAnimationFrame(animate);

    if (car) {
        parkingSpaces.forEach(space => space.position.z += carSpeed);
        parkedCars.forEach(c => c.position.z += carSpeed);
        otherObjects.forEach(obj => obj.position.z += carSpeed);

        /*if (carWheels.length) {
            const rotationAngle = carSpeed / WHEEL_RADIUS;
            const axis = new THREE.Vector3(0, 0, 1);

            carWheels.forEach(wheel => {
                // rotate around itself (correct “spin”)
                wheel.position.applyAxisAngle(axis, rotationAngle);
                wheel.rotateOnAxis(axis, rotationAngle);
            });
        }*/

    }

    renderer.render(scene, camera);
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

    leftLight.position.set(4, 1.0, -2);
    rightLight.position.set(4, 1.0, 2);

    const leftTarget = new THREE.Object3D();
    const rightTarget = new THREE.Object3D();

    car.add(leftTarget);
    car.add(rightTarget);

    leftTarget.position.set(40, 0.6, 0.9);
    rightTarget.position.set(40, 0.6, -0.9);

    leftLight.target = leftTarget;
    rightLight.target = rightTarget;

    headlightGroup.add(leftLight);
    headlightGroup.add(rightLight);

    const bulbGeo = new THREE.SphereGeometry(0.08, 12, 12);
    const bulbMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const bulbL = new THREE.Mesh(bulbGeo, bulbMat);
    const bulbR = new THREE.Mesh(bulbGeo, bulbMat);
    bulbL.position.copy(leftLight.position);
    bulbR.position.copy(rightLight.position);
    headlightGroup.add(bulbL, bulbR);

    car.add(headlightGroup);

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
    envRT = new THREE.WebGLCubeRenderTarget(64);
    envCam = new THREE.CubeCamera(0.1, 500, envRT);
    scene.add(envCam);
    scene.environment = envRT.texture;

    // tone mapping – temna scena
    // ACES -> realisičen contrast in smooth highlights
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.55;

    // luna kot vizualni element
    const moonGeo = new THREE.SphereGeometry(5, 32, 32);
    const moonMat = new THREE.MeshBasicMaterial({
        color: 0xdde6ff,
        transparent: true,
        opacity: 0.85
    });
    moon = new THREE.Mesh(moonGeo, moonMat);
    moon.position.set(-70, 80, -100);
    scene.add(moon);
}

function addParkingGlow(space) {
    const light = new THREE.PointLight(
        0x00ff66, // vivid green
        2.5, // intensity (strong)
        7, // distance
        2  // decay
    );
    light.position.set(0, 0.8, 0);
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
        0xff00000,
        2.5,
        7,
        2
    );
    light.position.set(0, 0.8, 0);
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
    envRT = new THREE.WebGLCubeRenderTarget(128);
    envCam = new THREE.CubeCamera(0.1, 1000, envRT);
    scene.add(envCam);
    scene.environment = envRT.texture;

    // tone mapping – realen dan
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;

    // odstrani luno če obstaja
    if (moon) {
        scene.remove(moon);
        moon = null;
    }
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
});

function getCarWheels(car) {
    const wheels = [];

    car.traverse(child => {
        if (!child.isMesh) return;
        console.log(child);

        // adjust names if needed (console.log(child.name))
        if (
            child.name.toLowerCase().includes('wheel') ||
            child.name.toLowerCase().includes('tire') ||
            child.name.toLowerCase().includes('kolo')
        ) {
            wheels.push(child);
        }
    });

    console.log(wheels)
    return wheels;
}
