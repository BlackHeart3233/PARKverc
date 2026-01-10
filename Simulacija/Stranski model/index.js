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

    // LUČ
    // ambientna
    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    // direkcijska
    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.castShadow = true;
    light.position.set(-100, 100, 0);
    scene.add(light);

    // TLA
    ground = new THREE.Mesh(
        new THREE.BoxGeometry(100, 1, 1000),
        new THREE.MeshLambertMaterial({ color: 0xffaaaa })
    );
    ground.castShadow = true;
    ground.receiveShadow = true;
    ground.position.y = -2;
    scene.add(ground);

    // pred-naložimo avto
    preloadModel("objects-models/rac_grafika_model_armatura2.mtl", "objects-models/rac_grafika_model_armatura2.obj", (obj) => {
        carTemplate = obj;
        carTemplate.rotation.y += Math.PI / 2;

        // glavni avto
        car = cloneObject(carTemplate);
        car.position.set(0, 0, 0);
        scene.add(car);
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
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}