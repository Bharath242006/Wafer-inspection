import sys, os, csv, math
sys.path.append(os.path.abspath("."))
import cv2, numpy as np

def estimate_lattice_period_2d(ref_img: np.ndarray) -> tuple:
    """Estimates 2D lattice periods lambda_x, lambda_y in search image coordinates."""
    if ref_img.shape[0] > 200:
        ref_s = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_s = ref_img.copy()

    ref_f = ref_s.astype(np.float32) - np.mean(ref_s)
    f = np.fft.fft2(ref_f)
    power = np.abs(f)**2
    autocorr = np.real(np.fft.ifft2(power))
    autocorr = np.fft.fftshift(autocorr)

    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[cy-2:cy+3, cx-2:cx+3] = 0.0

    _, _, _, max_loc = cv2.minMaxLoc(autocorr)
    p_dx = max_loc[0] - cx
    p_dy = max_loc[1] - cy

    scale_fac = (ref_img.shape[0] / ref_s.shape[0]) * 10.0
    lx = abs(p_dx) * scale_fac if abs(p_dx) > 2 else 67.0
    ly = abs(p_dy) * scale_fac if abs(p_dy) > 2 else 67.0

    lx = float(np.clip(lx, 30.0, 150.0))
    ly = float(np.clip(ly, 30.0, 150.0))
    return lx, ly

def compute_sobel_grad(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    cv2.normalize(mag, mag, 0.0, 1.0, cv2.NORM_MINMAX)
    return mag

def zmuv_ncc(p1: np.ndarray, p2: np.ndarray) -> float:
    if p1.shape != p2.shape or p1.size == 0:
        return 0.0
    f1 = p1.astype(np.float32) - np.mean(p1)
    f2 = p2.astype(np.float32) - np.mean(p2)
    s1, s2 = np.std(f1), np.std(f2)
    if s1 > 1e-5 and s2 > 1e-5:
        return float(np.mean(f1 * f2) / (s1 * s2))
    return 0.0

def compute_ssim_simple(p1: np.ndarray, p2: np.ndarray) -> float:
    if p1.shape != p2.shape or p1.size == 0:
        return 0.0
    f1 = p1.astype(np.float32)
    f2 = p2.astype(np.float32)
    mu1 = np.mean(f1)
    mu2 = np.mean(f2)
    var1 = np.var(f1)
    var2 = np.var(f2)
    cov = np.mean((f1 - mu1) * (f2 - mu2))
    c1 = (0.01 * 255.0)**2
    c2 = (0.03 * 255.0)**2
    ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return float(ssim)

def run_periodic_identifiability_audit(num_samples=30):
    csv_path = os.path.join("dataset", "validation", "labels.csv")
    ref_dir = os.path.join("dataset", "validation", "reference")
    search_dir = os.path.join("dataset", "validation", "search")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "periodic_identifiability.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Label CSV not found at '{csv_path}'")
        return

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    audit_rows = []

    print("=" * 135)
    print("              CRITICAL DATASET PERIODIC IDENTIFIABILITY AUDIT (30 VALIDATION SAMPLES)")
    print("=" * 135)
    print(f"{'Sample ID':<10} {'Style':<7} {'True (x,y)':<18} {'Lattice (lx,ly)':<17} {'NCC Neigh X':<13} {'NCC Neigh Y':<13} {'Grad Sim':<10} {'SSIM':<8} {'Low-Freq Sim':<14} {'Identifiable?':<25}")
    print("-" * 135)

    identifiable_count = 0
    information_limited_count = 0

    for i in range(min(num_samples, len(records))):
        rec = records[i]
        filename = rec["image"]
        true_x = float(rec["x"])
        true_y = float(rec["y"])
        style = rec.get("style", "Unknown")

        ref_path = os.path.join(ref_dir, filename)
        search_path = os.path.join(search_dir, filename)

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        search_h, search_w = search_img.shape

        lx, ly = estimate_lattice_period_2d(ref_img)

        sw, sh = 100, 100
        tl_x = int(round(true_x - sw / 2.0))
        tl_y = int(round(true_y - sh / 2.0))

        # True target patch
        patch_target = search_img[max(0, tl_y):min(search_h, tl_y+sh), max(0, tl_x):min(search_w, tl_x+sw)]

        # Neighbor X patch (+lx px)
        tl_x_nx = int(round(true_x + lx - sw / 2.0))
        patch_nx = search_img[max(0, tl_y):min(search_h, tl_y+sh), max(0, tl_x_nx):min(search_w, tl_x_nx+sw)]

        # Neighbor Y patch (+ly px)
        tl_y_ny = int(round(true_y + ly - sh / 2.0))
        patch_ny = search_img[max(0, tl_y_ny):min(search_h, tl_y_ny+sh), max(0, tl_x):min(search_w, tl_x+sw)]

        # Pairwise similarities
        ncc_x = zmuv_ncc(patch_target, patch_nx)
        ncc_y = zmuv_ncc(patch_target, patch_ny)

        grad_target = compute_sobel_grad(patch_target)
        grad_nx = compute_sobel_grad(patch_nx)
        grad_sim = zmuv_ncc(grad_target, grad_nx)

        ssim_sim = compute_ssim_simple(patch_target, patch_nx)

        blur_target = cv2.GaussianBlur(patch_target.astype(np.float32), (21, 21), 5.0)
        blur_nx = cv2.GaussianBlur(patch_nx.astype(np.float32), (21, 21), 5.0)
        low_freq_sim = zmuv_ncc(blur_target, blur_nx)

        # Identifiability threshold check
        is_identifiable = not (ncc_x > 0.85 or ssim_sim > 0.80 or abs(ncc_x - 1.0) < 0.10)

        if is_identifiable:
            identifiable_count += 1
            ident_status = "True (Identifiable)"
        else:
            information_limited_count += 1
            ident_status = "False (Info-Limited)"

        print(f"{filename:<10} {style:<7} ({true_x:.1f},{true_y:.1f})      ({lx:.1f},{ly:.1f})        {ncc_x:<13.4f} {ncc_y:<13.4f} {grad_sim:<10.4f} {ssim_sim:<8.4f} {low_freq_sim:<14.4f} {ident_status:<25}")

        audit_rows.append({
            "sample_id": filename,
            "architecture": style,
            "true_x": f"{true_x:.2f}",
            "true_y": f"{true_y:.2f}",
            "lattice_x": f"{lx:.2f}",
            "lattice_y": f"{ly:.2f}",
            "ncc_neighbor_x": f"{ncc_x:.4f}",
            "ncc_neighbor_y": f"{ncc_y:.4f}",
            "gradient_similarity": f"{grad_sim:.4f}",
            "ssim_similarity": f"{ssim_sim:.4f}",
            "low_frequency_similarity": f"{low_freq_sim:.4f}",
            "identifiable": is_identifiable
        })

    # Save to results/periodic_identifiability.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "architecture", "true_x", "true_y", "lattice_x", "lattice_y",
            "ncc_neighbor_x", "ncc_neighbor_y", "gradient_similarity", "ssim_similarity",
            "low_frequency_similarity", "identifiable"
        ])
        writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)

    print("=" * 135)
    print(f"AUDIT SUMMARY:")
    print(f"  Total Samples Audited:          {len(audit_rows)}")
    print(f"  Unique Local Targets:           {identifiable_count} ({identifiable_count/len(audit_rows)*100.0:.1f}%)")
    print(f"  Information-Limited Aliases:    {information_limited_count} ({information_limited_count/len(audit_rows)*100.0:.1f}%)")
    print(f"Saved detailed audit report to:   '{out_csv}'")
    print("=" * 135)

if __name__ == "__main__":
    run_periodic_identifiability_audit(30)
