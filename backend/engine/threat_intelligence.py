import numpy as np
import joblib
import os
import re
import torch
import csv
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd

# Production artifacts live in <repo>/Execution; resolve them from this file so the
# engine finds them no matter which directory the server is launched from.
EXECUTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Execution")
MODEL_PATH = os.path.join(EXECUTION_DIR, "risk_model.pkl")
ENCODER_PATH = os.path.join(EXECUTION_DIR, "label_encoders.pkl")
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
            print("CRITICAL: Production models missing. Running in deterministic fallback mode.")
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

        print("Supplychainer V3 Brain Loaded: Production-Ready.")

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
                "Fog and snowstorm ground flights and close highways.",
                "Blizzard, snow and ice close roads and halt freight trains.",
                "Drought and low water levels restrict ships and barges on the waterway.",
                "Extreme heat and wildfires damage rail lines and close roads."],
    "labour": ["Dock workers and truck drivers go on strike.",
               "Union walkout halts terminal operations.",
               "Labour dispute causes a work stoppage at the port.",
               "Rail workers strike and freight trains stop running.",
               "Airport cargo handlers and air traffic controllers walk out.",
               "Employer lockout shuts down the rail network.",
               "Drivers blockade roads and depots in protest."],
    "geopolitical": ["Military conflict and missile attacks threaten commercial shipping.",
                     "Sanctions and a trade embargo block cargo.",
                     "Naval blockade closes the strait to vessels.",
                     "Armed forces or pirates seize and board a merchant ship.",
                     "War and shelling hit a port and shipping lines suspend calls.",
                     "Airspace closed by conflict forces flights to reroute.",
                     "Government export ban and border closure halt trade."],
    "infrastructure": ["Container ship runs aground and blocks the canal.",
                       "Bridge collapse and a train derailment cut the route.",
                       "Crane failure and a power outage stop the terminal.",
                       "Explosion and fire destroy port facilities and warehouses.",
                       "Fire on board a ship forces it to divert.",
                       "Runway or tunnel closure after an accident blocks traffic.",
                       "Lock or canal gate failure closes the waterway."],
    "cyber": ["Cyberattack and ransomware shut down port IT systems.",
              "Hackers attack booking and scheduling systems, halting operations.",
              "Malware disrupts logistics company networks and signalling."],
    "congestion": ["Severe congestion and long vessel queues at the port.",
                   "Container backlog and yard congestion delay cargo.",
                   "Ships wait at anchor for days to reach a berth.",
                   "Trucks queue for days at a congested border crossing.",
                   "Air cargo piles up at the airport as handling capacity runs short.",
                   "Rail yards are overwhelmed and freight trains back up.",
                   "Equipment shortage slows cargo pickups."],
}
MIN_TYPE_SIMILARITY = 0.30  # below this the report matches no category well
# The type is learned (logistic regression on the embeddings) from these
# archetypes plus the labelled disruptions in ml/nlp_headlines.csv. Only the dev
# and test splits are used; the holdout split stays out so it can measure the
# result. Cross-split accuracy on dev/test: 0.95, vs 0.90 for the nearest archetype.
TYPE_EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml", "nlp_headlines.csv")
TYPE_REGULARISATION = 4.0  # accuracy was flat from C=1 to C=32

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

# The historical safe corpus only covers routine operations. Most logistics
# headlines are routine business news, and news that a disruption has ended
# shares its words ("strike", "congestion", "storm"); without archetypes for
# both, they read as threats.
SAFE_ARCHETYPES = [
    "Port opens a new terminal and expands handling capacity.",
    "Logistics company reports strong quarterly earnings and revenue growth.",
    "Cargo volumes hit a record high as trade flows smoothly.",
    "Carrier launches a new weekly service and invests in new vessels.",
    "Company announces a new product, investment or appointment.",
    "Company launches a sustainability or zero-emission programme.",
    "Freight rates, prices and shipping stocks move with market demand.",
    "New infrastructure opens and cuts journey times.",
    "Industry award, anniversary celebration or trade conference.",
    "Safety drill or inspection completed with no problems found.",
    "Forecast calls for calm weather this season.",
    "Disruption ends and operations return to normal.",
    "Strike is called off after unions and employers reach an agreement.",
    "Congestion eases as queues clear and waiting times fall.",
    "Traffic resumes after the route reopens.",
]
# A headline is compared with the mean of its two closest safe anchors, so a
# single topically close one ("Congestion eases...") cannot cancel a real threat.
SAFE_TOP_K = 2

MIN_REPORT_WORDS = 4  # shorter fragments ("Light rain forecast") carry no reliable signal

# BAAI/bge-small-en-v1.5 (MIT, ~130 MB). On the dev split of ml/nlp_headlines.csv
# it separated disruptions from routine news better than all-MiniLM-L6-v2 with
# the same anchors (AUC 0.987 vs 0.954); see docs/MODEL_CARD.md.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOWNLOAD_ATTEMPTS = 3  # the Hugging Face Hub drops connections now and then

def split_reports(news_text: str) -> List[str]:
    """One chunk per headline or sentence, so unrelated headlines are not blended."""
    parts = re.split(r"\s+\|\s+|(?<=[.!?])\s+", news_text or "")
    return [p.strip()[:256] for p in parts if len(p.split()) >= MIN_REPORT_WORDS]

class ContrastiveNLPEngine:
    """Stage 2: PRODUCTION Contrastive NLP Brain."""
    def __init__(self, lazy_load=False):
        self._ready = False
        # Threat margin = best cosine similarity to a disaster anchor minus the mean
        # of the two best to safe anchors, per headline. The floor is set on the dev
        # split of ml/nlp_headlines.csv: above 0.08 it caught 98% of disruptions
        # with no false alarm. A full canal closure reaches ~0.34, strikes and
        # attacks ~0.17, routine congestion ~0.13, so the ramp to 0.35 keeps a
        # closure near 1 and routine congestion low. The score feeds the delay
        # model as incident severity.
        self.noise_floor = 0.08
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
                self.model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                # First run: download, retrying with a short backoff.
                for attempt in range(DOWNLOAD_ATTEMPTS):
                    try:
                        self.model = SentenceTransformer(EMBEDDING_MODEL)
                        break
                    except Exception as e:
                        if attempt == DOWNLOAD_ATTEMPTS - 1:
                            raise
                        print(f"[NLP ENGINE] Download failed ({e}); retrying...")
                        time.sleep(2 ** attempt)
            self.util = util
            pairs = [(t, s) for t, sentences in THREAT_TYPE_ANCHORS.items() for s in sentences]
            self._type_names = [t for t, _ in pairs]
            self._type_matrix = self.model.encode([s for _, s in pairs], convert_to_tensor=True)
            self._type_model = self._fit_type_model(pairs)
            # Historical incidents plus the category archetypes as disaster anchors;
            # routine operations plus positive business news as safe ones. Encoded
            # from text at start-up, so the anchors stay readable and auditable.
            self.disaster_matrix = torch.cat([self.model.encode(HISTORICAL_DISASTERS, convert_to_tensor=True),
                                              self._type_matrix])
            self.safe_matrix = self.model.encode(HISTORICAL_SAFE + SAFE_ARCHETYPES, convert_to_tensor=True)
            self._ready = True
            print("NLP Brain: Anchor matrices ready.")
        except Exception as e:
            print(f"[NLP ENGINE] Warmup failed: {e}")
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def assess(self, news_text: str) -> Dict[str, Any]:
        """Threat score and type of a report, both taken from its most threatening headline.

        Each headline is compared with its own best safe match, so an unrelated
        headline can neither dilute the threat nor lend it a different type.
        """
        chunks = split_reports(news_text)
        if not self._ready or not chunks:
            return {"score": 0.0, "margin": 0.0, "type": "none", "confidence": 0.0, "headline": None}
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        d_scores = self.util.cos_sim(embeddings, self.disaster_matrix).cpu().numpy().max(axis=1)
        safe_sims = np.sort(self.util.cos_sim(embeddings, self.safe_matrix).cpu().numpy(), axis=1)
        s_scores = safe_sims[:, -SAFE_TOP_K:].mean(axis=1)
        margins = d_scores - s_scores
        top = int(np.argmax(margins))
        margin = float(margins[top])
        score = 0.0 if margin <= self.noise_floor else float(
            min(1.0, (margin - self.noise_floor) / (self.saturation_margin - self.noise_floor)))
        similarity = self.util.cos_sim(embeddings[top:top + 1], self._type_matrix).cpu().numpy()[0]
        probabilities = self._type_model.predict_proba(embeddings[top:top + 1].cpu().numpy())[0]
        best = int(np.argmax(probabilities))
        threat_type = self._type_model.classes_[best] if similarity.max() >= MIN_TYPE_SIMILARITY else "general"
        return {"score": score, "margin": round(margin, 4), "type": str(threat_type),
                "confidence": round(float(probabilities[best]), 3), "headline": chunks[top]}

    def _fit_type_model(self, pairs):
        """Threat-type classifier over the archetypes and the labelled dev/test disruptions."""
        from sklearn.linear_model import LogisticRegression
        texts, labels = [s for _, s in pairs], [t for t, _ in pairs]
        if os.path.exists(TYPE_EXAMPLES):
            with open(TYPE_EXAMPLES, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["label"] == "disrupted" and row["split"] in ("dev", "test"):
                        texts.append(row["headline"])
                        labels.append(row["threat_type"])
        return LogisticRegression(C=TYPE_REGULARISATION, max_iter=4000).fit(self.model.encode(texts), labels)

    def get_semantic_score(self, news_text: str) -> float:
        return self.assess(news_text)["score"]

    def classify_threat(self, news_text: str) -> Optional[Dict[str, Any]]:
        """Disruption category of a report's most threatening headline."""
        if not self._ready or not split_reports(news_text): return None
        result = self.assess(news_text)
        return {"type": result["type"], "confidence": result["confidence"]}

class CARFFilter:
    """Stage 3: CARF (Context-Aware Relevance Filter).

    A threat is dropped when the news is clearly about a different transport mode:
    it names another mode's infrastructure and none of this leg's own. News that
    names no mode (weather, conflict, cyberattacks) stays relevant to every mode.
    """
    def __init__(self):
        # Only words that name one mode's infrastructure or workforce. Generic words
        # such as "station" (weather station), "track" (track a storm), "bridge"
        # (road or rail), "terminal" and "container" (every mode), "freighter"
        # (ship or cargo plane), "docker" (also software) or "anchorage" (also an
        # air-cargo city) would misfile news and wrongly drop it for other modes.
        self.relevance_map = {
            "air": {"airport", "flight", "airspace", "aviation", "airline", "aircraft", "runway", "airfield"},
            "sea": {"port", "seaport", "vessel", "ship", "shipping", "canal", "ocean", "maritime",
                    "dock", "berth", "berthing", "harbor", "harbour", "strait", "tanker",
                    "dockworker", "dockers", "longshore", "longshoreman", "longshoremen", "stevedore",
                    "wharf", "wharfie", "quay", "pier", "barge", "waterway", "seafarer"},
            "rail": {"rail", "railway", "railroad", "locomotive", "train", "derailment", "wagon"},
            "road": {"highway", "motorway", "truck", "trucker", "trucking", "lorry", "road",
                     "haulier", "haulage", "interstate", "expressway", "freeway", "autobahn", "drayage"},
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
