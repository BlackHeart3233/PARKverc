import time
import cv2
import numpy as np

from compression import stisni_sliko_bytes
from decompression import razsiri_sliko_bytes


IMAGE_PATH = "test.jpg"
FAKTOR_STISKANJA = 10
ITERATIONS = 3


def ts():
    """Readable timestamp"""
    return time.perf_counter()


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Could not load image")

    h, w, _ = img.shape
    print(f"Loaded image: {w}x{h}\n")

    # warm-up
    print("Warm-up run...")
    t0 = ts()
    data = stisni_sliko_bytes(img, FAKTOR_STISKANJA)
    t1 = ts()
    out = razsiri_sliko_bytes(data)
    t2 = ts()
    print(f"  Warm-up compression: {(t1 - t0)*1000:.2f} ms")
    print(f"  Warm-up decompression: {(t2 - t1)*1000:.2f} ms\n")

    comp_times = []
    decomp_times = []

    for i in range(ITERATIONS):
        print(f"Iteration {i+1}/{ITERATIONS}")

        t_start = ts()
        print(f"  Compression start: {t_start:.6f}")

        data = stisni_sliko_bytes(img, FAKTOR_STISKANJA)

        t_comp_end = ts()
        print(f"  Compression end:   {t_comp_end:.6f}")
        print(f"  Compression time:  {(t_comp_end - t_start)*1000:.2f} ms")

        t_decomp_start = ts()
        print(f"  Decompression start: {t_decomp_start:.6f}")

        out = razsiri_sliko_bytes(data)

        t_decomp_end = ts()
        print(f"  Decompression end:   {t_decomp_end:.6f}")
        print(f"  Decompression time:  {(t_decomp_end - t_decomp_start)*1000:.2f} ms\n")

        comp_times.append(t_comp_end - t_start)
        decomp_times.append(t_decomp_end - t_decomp_start)

    print("--- Summary ---")
    print(f"Compressed size: {len(data)} bytes")
    print(f"Compression avg: {np.mean(comp_times)*1000:.2f} ms")
    print(f"Decompression avg: {np.mean(decomp_times)*1000:.2f} ms")
    print(f"Total avg: {(np.mean(comp_times)+np.mean(decomp_times))*1000:.2f} ms")

    combined = np.hstack([img, out])
    cv2.imshow("Original (left) | Reconstructed (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
