"""
Enriched Feature extraction for Indian Sign Language (ISL) alphabet recognition.
Converts 126-D MediaPipe landmark coordinates into a 202-D geometric invariant feature vector:
  - 126 normalized Cartesian coordinates (wrist-centered and hand-span scaled)
  - 18 Left-hand geometric invariants (tip-wrist, tip-thumb, adjacent tip spans, finger curls, pinch ratio)
  - 18 Right-hand geometric invariants
  - 10 Left-hand knuckle joint bending angles (MCP-PIP-DIP & PIP-DIP-TIP cosines)
  - 10 Right-hand knuckle joint bending angles
  - 3 Left-hand palm surface normal orientation coordinates (wrist roll & tilt)
  - 3 Right-hand palm surface normal orientation coordinates
  - 14 Inter-hand cross-interaction distances (critical for contact signs like M, N, T, K, P, R, S)
"""

import numpy as np

NUM_RAW_FEATURES = 126
NUM_EXTRACTED_FEATURES = 202

# Joint triplets for angle calculation (MCP, PIP, DIP) and (PIP, DIP, TIP)
FINGER_JOINT_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),       # Thumb
    (5, 6, 7), (6, 7, 8),       # Index
    (9, 10, 11), (10, 11, 12),  # Middle
    (13, 14, 15), (14, 15, 16), # Ring
    (17, 18, 19), (18, 19, 20), # Pinky
]


def _compute_angle_cos(p_a, p_b, p_c):
    """Compute cosine of the 3D angle at vertex p_b between vectors (p_a - p_b) and (p_c - p_b)."""
    v1 = p_a - p_b
    v2 = p_c - p_b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 1.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(cos_theta, -1.0, 1.0))


def extract_features(raw_landmarks):
    """
    Extract a 202-D normalized geometric feature vector from 126 raw landmark floats.
    Handles single sample (126,) or batch (N, 126).
    """
    raw = np.asarray(raw_landmarks, dtype=np.float32)
    single = (raw.ndim == 1)
    if single:
        raw = raw[None, :]
    
    N = len(raw)
    pts = raw.reshape(N, 2, 21, 3)
    feats = []
    
    for i in range(N):
        p = pts[i]
        present = np.any(p != 0, axis=(1, 2))
        if not present.any():
            feats.append(np.zeros(NUM_EXTRACTED_FEATURES, dtype=np.float32))
            continue
            
        wrists = p[present, 0]
        anchor = wrists.mean(axis=0)
        hand_sizes = np.linalg.norm(p[present, 12] - p[present, 0], axis=1)
        scale = float(hand_sizes.mean())
        if not np.isfinite(scale) or scale < 1e-4:
            scale = 1.0
        
        norm_p = (p - anchor) / scale
        norm_p[~present] = 0
        norm_coords = norm_p.reshape(-1)
        
        geo = []
        for h in range(2):
            if not present[h]:
                geo.extend([0.0] * (18 + 10 + 3))
                continue
            hp = norm_p[h]
            w = hp[0]
            thumb_tip, index_tip, mid_tip, ring_tip, pinky_tip = hp[4], hp[8], hp[12], hp[16], hp[20]
            mcp_t, mcp_i, mcp_m, mcp_r, mcp_p = hp[2], hp[5], hp[9], hp[13], hp[17]
            
            # 1. Tip-to-wrist (5)
            d_w_t = np.linalg.norm(thumb_tip - w)
            d_w_i = np.linalg.norm(index_tip - w)
            d_w_m = np.linalg.norm(mid_tip - w)
            d_w_r = np.linalg.norm(ring_tip - w)
            d_w_p = np.linalg.norm(pinky_tip - w)
            
            # 2. Tip-to-thumb (4)
            d_t_i = np.linalg.norm(index_tip - thumb_tip)
            d_t_m = np.linalg.norm(mid_tip - thumb_tip)
            d_t_r = np.linalg.norm(ring_tip - thumb_tip)
            d_t_p = np.linalg.norm(pinky_tip - thumb_tip)
            
            # 3. Adjacent fingertip (3)
            d_i_m = np.linalg.norm(mid_tip - index_tip)
            d_m_r = np.linalg.norm(ring_tip - mid_tip)
            d_r_p = np.linalg.norm(pinky_tip - ring_tip)
            
            # 4. Finger curl (Tip to MCP distance) (5)
            c_t = np.linalg.norm(thumb_tip - mcp_t)
            c_i = np.linalg.norm(index_tip - mcp_i)
            c_m = np.linalg.norm(mid_tip - mcp_m)
            c_r = np.linalg.norm(ring_tip - mcp_r)
            c_p = np.linalg.norm(pinky_tip - mcp_p)
            
            # 5. Pinch ratio (1)
            pinch = np.linalg.norm(thumb_tip - index_tip) / (d_w_i + 1e-6)
            
            geo.extend([d_w_t, d_w_i, d_w_m, d_w_r, d_w_p, d_t_i, d_t_m, d_t_r, d_t_p, d_i_m, d_m_r, d_r_p, c_t, c_i, c_m, c_r, c_p, pinch])

            # 6. Joint bending angles (10)
            for a, b, c in FINGER_JOINT_TRIPLETS:
                geo.append(_compute_angle_cos(hp[a], hp[b], hp[c]))

            # 7. Palm surface normal unit vector (3)
            v_idx = hp[5] - hp[0]
            v_pnk = hp[17] - hp[0]
            n_palm = np.cross(v_idx, v_pnk)
            norm_np = np.linalg.norm(n_palm)
            if norm_np > 1e-6:
                n_palm = n_palm / norm_np
            else:
                n_palm = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            geo.extend([float(n_palm[0]), float(n_palm[1]), float(n_palm[2])])
        
        # Inter-hand interactions (14)
        if present[0] and present[1]:
            d_h_w = np.linalg.norm(norm_p[0, 0] - norm_p[1, 0])
            d_idx_cross = np.linalg.norm(norm_p[0, 8] - norm_p[1, 8])
            d_t_cross = np.linalg.norm(norm_p[0, 4] - norm_p[1, 4])
            d_p_cross = np.linalg.norm(norm_p[0, 20] - norm_p[1, 20])
            
            # Right hand tips to Left Palm/Wrist
            r_i_lw = np.linalg.norm(norm_p[1, 8] - norm_p[0, 0])
            r_m_lw = np.linalg.norm(norm_p[1, 12] - norm_p[0, 0])
            r_r_lw = np.linalg.norm(norm_p[1, 16] - norm_p[0, 0])
            r_i_lp = np.linalg.norm(norm_p[1, 8] - norm_p[0, 9])
            r_m_lp = np.linalg.norm(norm_p[1, 12] - norm_p[0, 9])
            r_r_lp = np.linalg.norm(norm_p[1, 16] - norm_p[0, 9])
            
            # Left hand tips to Right Palm/Wrist
            l_i_rw = np.linalg.norm(norm_p[0, 8] - norm_p[1, 0])
            l_m_rw = np.linalg.norm(norm_p[0, 12] - norm_p[1, 0])
            l_i_rp = np.linalg.norm(norm_p[0, 8] - norm_p[1, 9])
            l_m_rp = np.linalg.norm(norm_p[0, 12] - norm_p[1, 9])
            
            geo.extend([d_h_w, d_idx_cross, d_t_cross, d_p_cross, r_i_lw, r_m_lw, r_r_lw, r_i_lp, r_m_lp, r_r_lp, l_i_rw, l_m_rw, l_i_rp, l_m_rp])
        else:
            geo.extend([0.0] * 14)
            
        full_vec = np.concatenate([norm_coords, np.array(geo, dtype=np.float32)])
        feats.append(full_vec)
        
    res = np.array(feats, dtype=np.float32)
    return res[0] if single else res
