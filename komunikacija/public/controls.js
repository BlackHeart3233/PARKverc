import * as THREE from './three.js';


export class CameraControls {
    constructor(camera, domElement, options = {}) {
        this.camera = camera;
        this.domElement = domElement;

        // kam gleda kamera
        this.target = (options.target || new THREE.Vector3(0, 0, 0)).clone();

        // konstante -> spremenite kak vam paše - tale men najbolj ustreza
        this.rotateSpeed = options.rotateSpeed || 0.005; // levi klik
        this.panSpeed = options.panSpeed || 0.02; // sredinski klik
        this.zoomSpeed = options.zoomSpeed || 0.15; // scroll
        this.minDistance = options.minDistance ?? 5;
        this.maxDistance = options.maxDistance ?? 150;

        this.state = 'none'; // 'none' | 'rotate' | 'pan'
        this.isMouseDown = false;
        this.prevMouse = {x: 0, y: 0};

        this.offset = new THREE.Vector3();
        this.spherical = new THREE.Spherical();
        this.updateSphericalFromCamera();


        this.onMouseDown = this.onMouseDown.bind(this);
        this.onMouseMove = this.onMouseMove.bind(this);
        this.onMouseUp = this.onMouseUp.bind(this);
        this.onWheelScroll = this.onWheelScroll.bind(this);

        this.domElement.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
        this.domElement.addEventListener('wheel', this.onWheelScroll, {passive: false});
    }

    dispose() {
        this.domElement.removeEventListener('mousedown', this.onMouseDown);
        window.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.domElement.removeEventListener('wheel', this.onWheelScroll);
    }

    updateSphericalFromCamera() {
        // offset = camera.position - target
        this.offset.subVectors(this.camera.position, this.target);
        this.spherical.setFromVector3(this.offset);
    }

    updateCameraFromSpherical() {
        // klamp na min-max razdaljo
        this.spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, this.spherical.radius));

        // spherical -> cartesian in update
        this.offset.setFromSpherical(this.spherical);
        this.camera.position.copy(this.target).add(this.offset);
        this.camera.lookAt(this.target);
    }

    onMouseDown(event) {
        // 0 = left,
        // 1 = middle
        // 2 = right
        if (event.button === 0) {
            this.state = 'rotate';
            this.isMouseDown = true;
        } else if (event.button === 1) {
            this.state = 'pan';
            this.isMouseDown = true;
        } else {
            return;
        }

        event.preventDefault();
        this.prevMouse.x = event.clientX;
        this.prevMouse.y = event.clientY;
    }

    onMouseMove(event) {
        if (!this.isMouseDown || this.state === 'none') return;

        const dx = event.clientX - this.prevMouse.x;
        const dy = event.clientY - this.prevMouse.y;

        if (this.state === 'rotate') {
            this.handleRotate(dx, dy);
        } else if (this.state === 'pan') {
            this.handlePan(dx, dy);
        }

        this.prevMouse.x = event.clientX;
        this.prevMouse.y = event.clientY;
    }

    onMouseUp(event) {
        if (event.button === 0 && this.state === 'rotate') {
            this.isMouseDown = false;
            this.state = 'none';
        } else if (event.button === 1 && this.state === 'pan') {
            this.isMouseDown = false;
            this.state = 'none';
        }
    }

    onWheelScroll(event) {
        event.preventDefault();

        // deltaY < 0 => scroll "gor" => zoom in
        // deltaY > 0 => scroll "dol" => zoom out
        const zoomFactor = this.zoomSpeed;

        if (event.deltaY < 0) {
            this.spherical.radius *= (1 - zoomFactor);
        } else if (event.deltaY > 0) {
            this.spherical.radius *= (1 + zoomFactor);
        }

        this.updateCameraFromSpherical();
    }


    handleRotate(dx, dy) {
        const rotateSpeed = this.rotateSpeed;

        this.spherical.theta -= dx * rotateSpeed; // left/right
        this.spherical.phi -= dy * rotateSpeed; // up/down

        const EPS = 0.000001;
        this.spherical.phi = Math.max(EPS, Math.min(Math.PI - EPS, this.spherical.phi));

        this.updateCameraFromSpherical();
    }

    // horizontalni premik basicaly
    handlePan(dx, dy) {
        const panSpeed = this.panSpeed;

        const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
        const distance = offset.length();

        // da ni prehitro se skalira z razdaljo
        const panScale = distance * panSpeed;

        const forward = new THREE.Vector3();
        this.camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(forward, this.camera.up).normalize();

        const panOffset = new THREE.Vector3();
        panOffset.addScaledVector(right, -dx * panScale);
        panOffset.addScaledVector(forward, dy * panScale);

        this.camera.position.add(panOffset);
        this.target.add(panOffset);

        this.updateSphericalFromCamera();
    }
}