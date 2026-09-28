import numpy as np
import joblib
import os
import re
import torch
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Production artifacts live in <repo>/Execution; resolve them from this file so the
# engine finds them no matter which directory the server is launched from.
EXECUTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Execution")
MODEL_PATH = os.path.join(EXECUTION_DIR, "risk_model.pkl")
ENCODER_PATH = os.path.join(EXECUTION_DIR, "label_encoders.pkl")
NLP_ANCHORS_PATH = os.path.join(EXECUTION_DIR, "nlp_anchors.pt")
CALIBRATION_PATH = os.path.join(EXECUTION_DIR, "calibration_profiles.json")

class ThreatIntelligencePredictor:
    """
    Supplychainer Quantile ML Decision Brain.
    V3: Statistically Defensible Calibration & Geographic Hub Intelligence.
    """
    def __init__(self, lazy_load=False):
        self.is_trained = False
        self.model = None
        self.encoders = None
        self.profiles = {}
        
        self.hub_map = {
            "Seattle": "Seattle Port", "Portland": "Portland Terminal", "San Francisco": "San Francisco Port",
            "Los Angeles": "Los Angeles Port", "Salt Lake City": "Salt Lake City Hub", "Denver": "Denver Terminal",
            "Phoenix": "Phoenix Logistics", "Dallas": "Dallas Corridor", "Houston": "Houston Port",
            "Chicago": "Chicago Rail Hub", "St. Louis": "St. Louis Hub", "Atlanta": "Atlanta Air Hub",
            "Miami": "Miami Port", "New York": "New York Port", "Boston": "Boston Terminal",
            "Mumbai": "Mumbai Port", "Kochi": "Kochi Port", "Delhi": "Delhi Air Cargo", "Chennai": "Chennai Port"
        }
        
        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self.is_trained: return
        print("[PREDICTOR] Starting warmup...")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            print(f"CRITICAL: Production models missing. Running in deterministic fallback mode.")
            return
            
        # 1. Load ML Core
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODER_PATH)
        self.is_trained = True
        
        # 2. Load Statistically Defensible Calibration Profiles
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH, 'r') as f:
                self.profiles = json.load(f)
            print(f"Calibration Layer: Loaded {len(self.profiles)} mode profiles from historical p5/p95 analysis.")
        else:
            print("WARNING: Calibration profiles missing. Using defensive fallbacks.")
            self.profiles = {}

        print(f"Supplychainer V3 Brain Loaded: Production-Ready.")

    def _encode_feature(self, value: str, key: str) -> int:
        encoder = self.encoders[key]
        classes = list(encoder.classes_)
        if key in ["Origin_Node", "Destination_Node"]:
            resolved = self.hub_map.get(value, value)
            if resolved in classes: return encoder.transform([resolved])[0]
        if value in classes: return encoder.transform([value])[0]
        return encoder.transform([classes[0]])[0]

    def predict_worst_case_delay(self, origin: str, destination: str, transport_mode: str, 
                                 leg_type: str = "Global_Freight", condition_flag: str = "Clear", 
                                 nlp_score: float = 0.0) -> Dict[str, Any]:
        """
        Stage 4: p85 Quantile Prediction with Statistically Defensible Calibration.
        """
        if not self.is_trained:
            mode_key = transport_mode.lower()
            priors = {"road": 2.5, "sea": 48.0, "air": 12.0, "rail": 18.0}
            delay = priors.get(mode_key, 12.0)
            return {
                "raw_model_prediction": delay,
                "calibrated_delay": delay,
                "baseline_systemic_friction": delay,
                "final_delay_presented": delay,
                "calibration_reason": "Deterministic Operational Prior (Engine Warming)",
                "p_quantile": 0.85,
                "is_defensible": True
            }

        # t_ml_start = time.perf_counter()
        try:
            feat_origin = self._encode_feature(origin, 'Origin_Node')
            feat_dest = self._encode_feature(destination, 'Destination_Node')
            feat_mode = self._encode_feature(transport_mode, 'Transport_Mode')
            feat_leg = self._encode_feature(leg_type, 'Leg_Type')
            feat_cond = self._encode_feature(condition_flag, 'Condition_Flag')
            
            X_input = pd.DataFrame([{'Leg_Type': feat_leg, 'Origin_Node': feat_origin, 'Destination_Node': feat_dest,
                                     'Transport_Mode': feat_mode, 'Condition_Flag': feat_cond, 'NLP_Severity_Score': nlp_score}])
            
            # 1. Raw p85 Inference
            raw_prediction = float(self.model.predict(X_input)[0])
            
            # 2. Statistical Calibration (Derived from Historical p95)
            mode_key = transport_mode.lower()
            profile = self.profiles.get(mode_key, {"floor": 0.0, "cap": 240.0})
            
            floor = profile["floor"]
            cap = profile["cap"]
            
            calibrated_delay = min(max(0.0, raw_prediction), cap)
            
            # 3. Restore Systemic Friction (p5 Baseline)
            final_delay = max(calibrated_delay, floor)
            
            # Explainability
            reason = "Optimal Flow"
            if final_delay == floor and calibrated_delay < floor:
                reason = f"Baseline Operational Friction (Historical p5: {floor}h)"
            elif calibrated_delay < raw_prediction:
                reason = f"Operational Cap Applied (Historical p95 Bound: {cap}h)"
            elif raw_prediction > floor:
                reason = "Quantile Disruption Prediction (p85 Risk)"

            return {
                "raw_model_prediction": round(raw_prediction, 2),
                "calibrated_delay": round(calibrated_delay, 2),
                "baseline_systemic_friction": floor,
                "final_delay_presented": round(final_delay, 2),
                "calibration_reason": reason,
                "p_quantile": 0.85,
                "is_defensible": True
            }
            
        except Exception as e:
            print(f"Calibration Inference Error: {e}")
            return {"final_delay_presented": 0.0, "calibration_reason": "Inference Error"}

class ContrastiveNLPEngine:
    """Stage 2: PRODUCTION Contrastive NLP Brain."""
    def __init__(self, lazy_load=False):
        self._ready = False
        # Threat Margin = max cos-sim to a disaster anchor minus max cos-sim to a
        # safe anchor. Measured against the anchor corpus: routine/safe text sits
        # below 0.08, operational disruptions (strikes, floods) around 0.2-0.3, and
        # full closures (Suez grounding, Red Sea attacks) at 0.5+. The score ramps
        # linearly between the noise floor and the saturation point.
        self.noise_floor = 0.08
        self.saturation_margin = 0.50
        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self._ready: return
        print("[NLP ENGINE] Starting warmup...")
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.util = util
            if os.path.exists(NLP_ANCHORS_PATH):
                # The anchors were saved from a GPU session; map them to CPU so the
                # engine also runs on machines without CUDA.
                anchors = torch.load(NLP_ANCHORS_PATH, map_location="cpu", weights_only=True)
                self.disaster_matrix = anchors["disaster_matrix"]
                self.safe_matrix = anchors["safe_matrix"]
                self._ready = True
                print(f"NLP Brain: Loaded Historical Anchor Matrix.")
            else:
                self._ready = False
        except Exception as e:
            print(f"[NLP ENGINE] Warmup failed: {e}")
            self._ready = False

    def get_semantic_score(self, news_text: str) -> float:
        # t_nlp_start = time.perf_counter()
        if not self._ready: return 0.0
        if not news_text or len(news_text.strip()) < 5: return 0.0
        chunks = [news_text[i:i+256] for i in range(0, len(news_text), 256)]
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        d_scores = self.util.cos_sim(chunk_embeddings, self.disaster_matrix)
        s_scores = self.util.cos_sim(chunk_embeddings, self.safe_matrix)
        margin = float(np.max(d_scores.cpu().numpy())) - float(np.max(s_scores.cpu().numpy()))
        if margin <= self.noise_floor: return 0.0
        return float(min(1.0, (margin - self.noise_floor) / (self.saturation_margin - self.noise_floor)))

class CARFFilter:
    """Stage 3: CARF (Context-Aware Relevance Filter).

    A threat is dropped when the news is clearly about a different transport mode:
    it names another mode's infrastructure and none of this leg's own. News that
    names no mode (weather, conflict, cyberattacks) stays relevant to every mode.
    """
    def __init__(self):
        self.relevance_map = {
            "air": {"airport", "flight", "airspace", "aviation", "airline", "aircraft"},
            "sea": {"port", "seaport", "vessel", "ship", "shipping", "canal", "ocean", "maritime",
                    "dock", "berth", "berthing", "harbor", "harbour", "strait", "tanker"},
            "rail": {"rail", "railway", "railroad", "track", "locomotive", "train", "station", "derailment"},
            "road": {"highway", "motorway", "truck", "trucker", "lorry", "bridge", "road", "delivery"},
        }

    def modes_mentioned(self, news_context: str) -> set:
        tokens = set(re.findall(r"[a-z]+", (news_context or "").lower()))
        tokens |= {t[:-1] for t in tokens if t.endswith("s")}
        return {mode for mode, keywords in self.relevance_map.items() if tokens & keywords}

    def apply_filter(self, semantic_score: float, news_context: str, transport_mode: str) -> float:
        if semantic_score <= 0: return 0.0
        mentioned = self.modes_mentioned(news_context)
        if mentioned and transport_mode in self.relevance_map and transport_mode not in mentioned:
            return 0.0
        return semantic_score

    def max_pool_threats(self, scores: List[float]) -> float:
        return float(np.max(scores)) if scores else 0.0
