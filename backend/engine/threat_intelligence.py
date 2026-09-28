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

# Anchor sentences per disruption category, for zero-shot threat typing with the
# same sentence embedding used for the threat score.
THREAT_TYPE_ANCHORS = {
    "weather": ["Typhoon, hurricane or cyclone forces the port to close.",
                "Severe storm, heavy rain and flooding disrupt transport.",
                "Fog and snowstorm ground flights and close highways."],
    "labour": ["Dock workers and truck drivers go on strike.",
               "Union walkout halts terminal operations.",
               "Labour dispute causes a work stoppage at the port."],
    "geopolitical": ["Military conflict and missile attacks threaten commercial shipping.",
                     "Sanctions and a trade embargo block cargo.",
                     "Naval blockade closes the strait to vessels."],
    "infrastructure": ["Container ship runs aground and blocks the canal.",
                       "Bridge collapse and a train derailment cut the route.",
                       "Crane failure and a power outage stop the terminal."],
    "cyber": ["Cyberattack and ransomware shut down port IT systems."],
    "congestion": ["Severe congestion and long vessel queues at the port.",
                   "Container backlog and yard congestion delay cargo."],
}
MIN_TYPE_SIMILARITY = 0.30  # below this the report matches no category well

# Historical anchor corpus from Code/precompute_nlp.py with place and company names
# removed. The named originals leaked location into the score: every report about
# Rotterdam resembled the "Port of Rotterdam operating normally" safe anchor, so
# "Strike halts Rotterdam port operations" scored 0.
HISTORICAL_DISASTERS = [
    "A container vessel ran aground in a major canal, blocking all traffic in both directions for six days. Over 400 ships were delayed, causing billions in trade disruption.",
    "Ongoing security threats and missile attacks on commercial vessels have forced major shipping lines to reroute around a distant cape, adding 10-14 days to transit times.",
    "Port workers went on strike for 13 days, freezing a quarter of the country's traded goods and causing a massive backlog in rail and trucking networks.",
    "A potential nationwide rail strike threatened to shut down the freight network, risking $2 billion a day in economic output before emergency legislation was passed.",
    "A global ransomware attack disabled the IT systems of the world's largest shipping company, forcing manual operations at 76 port terminals worldwide.",
    "One of the world's busiest ports was partially shut down after a single COVID-19 case, causing severe global supply chain bottlenecks.",
    "Severe shortages of truck drivers led to fuel delivery failures and empty supermarket shelves, highlighting systemic vulnerability in road freight.",
    "Significant berthing congestion reported at container terminals. Vessel turnaround times are increasing due to labor shortages and yard density issues.",
    "Customs IT systems are experiencing intermittent connectivity, leading to manual processing and 48-hour backlogs for international freight.",
    "Trucker strikes and highway blockades have caused significant delays in last-mile delivery corridors. Port gates are experiencing high queue times.",
    "Severe shortages of storage space at major logistics hubs are causing dwell time penalties and secondary transport delays.",
    "Changes in regulatory inspections have created a bottleneck at the border, slowing down the flow of high-value cargo by 30%.",
    "Catastrophic flooding destroyed over 3,000 km of road network and damaged major rail bridges, halting all inland logistics for weeks.",
]
HISTORICAL_SAFE = [
    "Operations at the port are proceeding normally. Vessel turnaround times are within expected parameters and terminal capacity remains optimal.",
    "Standardized terminal operating procedures have achieved a 99% on-time departure rate. Efficiency gains in ground handling have reduced idle times.",
    "The freight transportation services index shows steady month-on-month growth. Intermodal rail volumes remain stable with no reported disruptions.",
    "Shipment cleared customs in 4 hours. No significant weather events reported on the transcontinental route. Traffic flowing at 100% capacity.",
    "Air cargo capacity on the transoceanic corridor remains high. Ground handling operations are normalized with no reported backlogs at major hubs.",
]

# The historical safe corpus only covers routine operations. Positive business
# news (new capacity, earnings, record volumes, new services) is common in
# logistics headlines and would otherwise read as mildly threatening.
SAFE_ARCHETYPES = [
    "Port opens a new terminal and expands handling capacity.",
    "Logistics company reports strong quarterly earnings and revenue growth.",
    "Cargo volumes hit a record high as trade flows smoothly.",
    "Carrier launches a new weekly service and invests in new vessels.",
]

def split_reports(news_text: str) -> List[str]:
    """One chunk per headline or sentence, so unrelated headlines are not blended."""
    parts = re.split(r"\s+\|\s+|(?<=[.!?])\s+", news_text or "")
    return [p.strip()[:256] for p in parts if len(p.strip()) >= 5]

class ContrastiveNLPEngine:
    """Stage 2: PRODUCTION Contrastive NLP Brain."""
    def __init__(self, lazy_load=False):
        self._ready = False
        # Threat Margin = max cos-sim to a disaster anchor minus max cos-sim to a
        # safe anchor, per headline. On a labelled set of 33 headlines, safe and
        # positive news stays below 0.10 and disruptions median ~0.2 (AUC 0.995).
        # The margin says how clearly a report describes a disruption; it is not a
        # fine-grained severity scale. The score ramps linearly from the noise
        # floor to full strength at a typical severe-disruption margin.
        self.noise_floor = 0.10
        self.saturation_margin = 0.35
        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self._ready: return
        print("[NLP ENGINE] Starting warmup...")
        try:
            from sentence_transformers import SentenceTransformer, util
            try:
                # Use the cached model without contacting the Hugging Face Hub, so
                # start-up works offline and on flaky networks.
                self.model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")  # first run: download
            self.util = util
            pairs = [(t, s) for t, sentences in THREAT_TYPE_ANCHORS.items() for s in sentences]
            self._type_names = [t for t, _ in pairs]
            self._type_matrix = self.model.encode([s for _, s in pairs], convert_to_tensor=True)
            # Historical incidents plus the category archetypes as disaster anchors;
            # routine operations plus positive business news as safe ones. Encoded
            # from text at start-up, so the anchors stay readable and auditable.
            self.disaster_matrix = torch.cat([self.model.encode(HISTORICAL_DISASTERS, convert_to_tensor=True),
                                              self._type_matrix])
            self.safe_matrix = self.model.encode(HISTORICAL_SAFE + SAFE_ARCHETYPES, convert_to_tensor=True)
            self._ready = True
            print(f"NLP Brain: Anchor matrices ready.")
        except Exception as e:
            print(f"[NLP ENGINE] Warmup failed: {e}")
            self._ready = False

    def get_semantic_score(self, news_text: str) -> float:
        # t_nlp_start = time.perf_counter()
        if not self._ready: return 0.0
        chunks = split_reports(news_text)
        if not chunks: return 0.0
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        d_scores = self.util.cos_sim(chunk_embeddings, self.disaster_matrix).cpu().numpy().max(axis=1)
        s_scores = self.util.cos_sim(chunk_embeddings, self.safe_matrix).cpu().numpy().max(axis=1)
        # The most threatening headline decides; each is compared with its own best safe match.
        margin = float(np.max(d_scores - s_scores))
        if margin <= self.noise_floor: return 0.0
        return float(min(1.0, (margin - self.noise_floor) / (self.saturation_margin - self.noise_floor)))

    def classify_threat(self, news_text: str) -> Optional[Dict[str, Any]]:
        """Most likely disruption category of a report, with its anchor similarity."""
        chunks = split_reports(news_text)
        if not self._ready or not chunks: return None
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        similarity = self.util.cos_sim(embeddings, self._type_matrix).cpu().numpy().max(axis=0)
        best = int(np.argmax(similarity))
        threat_type = self._type_names[best] if similarity[best] >= MIN_TYPE_SIMILARITY else "general"
        return {"type": threat_type, "confidence": round(float(similarity[best]), 3)}

class CARFFilter:
    """Stage 3: CARF (Context-Aware Relevance Filter).

    A threat is dropped when the news is clearly about a different transport mode:
    it names another mode's infrastructure and none of this leg's own. News that
    names no mode (weather, conflict, cyberattacks) stays relevant to every mode.
    """
    def __init__(self):
        # Only words that name one mode's infrastructure. Generic words such as
        # "station" (weather station), "track" (track a storm) or "bridge" (road or
        # rail) would misfile mode-neutral news and wrongly drop it for other modes.
        self.relevance_map = {
            "air": {"airport", "flight", "airspace", "aviation", "airline", "aircraft"},
            "sea": {"port", "seaport", "vessel", "ship", "shipping", "canal", "ocean", "maritime",
                    "dock", "berth", "berthing", "harbor", "harbour", "strait", "tanker"},
            "rail": {"rail", "railway", "railroad", "locomotive", "train", "derailment"},
            "road": {"highway", "motorway", "truck", "trucker", "trucking", "lorry", "road"},
        }

    def modes_mentioned(self, news_context: str) -> set:
        tokens = set(re.findall(r"[a-z]+", (news_context or "").lower()))
        tokens |= {t[:-3] + "y" for t in tokens if t.endswith("ies")}  # lorries -> lorry
        tokens |= {t[:-1] for t in tokens if t.endswith("s")}         # vessels -> vessel
        return {mode for mode, keywords in self.relevance_map.items() if tokens & keywords}

    def apply_filter(self, semantic_score: float, news_context: str, transport_mode: str) -> float:
        if semantic_score <= 0: return 0.0
        mentioned = self.modes_mentioned(news_context)
        if mentioned and transport_mode in self.relevance_map and transport_mode not in mentioned:
            return 0.0
        return semantic_score

    def max_pool_threats(self, scores: List[float]) -> float:
        return float(np.max(scores)) if scores else 0.0
