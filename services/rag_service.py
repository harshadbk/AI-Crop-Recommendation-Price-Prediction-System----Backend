import os
import json
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.core.config import settings

class RAGService:
    def __init__(self):
        try:
            print(f"Connecting to Weaviate at {settings.WEAVIATE_URL}...")
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.WEAVIATE_URL,
                auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
            )
            print("Weaviate connection established.")
        except Exception as e:
            print(f"Weaviate Connection Failed: {e}")
            self.client = None
        
        # Initialize Embeddings
        print("Initializing HuggingFace Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL
        )

    def ingest_csv_data(self, file_path, collection_name):
        """Ingests CSV data into Weaviate V4 with batching"""
        if not self.client:
            return

        try:
            # Check if collection exists and has data
            if self.client.collections.exists(collection_name):
                collection = self.client.collections.get(collection_name)
                # Quick check for items
                response = collection.query.fetch_objects(limit=1)
                if len(response.objects) > 0:
                    print(f"Collection {collection_name} already initialized.")
                    return
            
            print(f"Creating/Ingesting data into {collection_name} from {file_path}...")
            df = pd.read_csv(file_path).head(2000) 
            
            # Combine columns into a single string for better embedding context
            # Use descriptive keys for better semantic search
            texts = []
            for _, row in df.iterrows():
                if collection_name == "MandiPrices":
                    text = f"In {row.get('State', 'N/A')}, district {row.get('District', 'N/A')}, the commodity {row.get('Commodity', 'N/A')} (variety {row.get('Variety', 'N/A')}) was traded at market {row.get('Market', 'N/A')} on {row.get('Arrival_Date', 'N/A')}. The modal price was ₹{row.get('Modal_Price', 0)} per quintal, with a range of ₹{row.get('Min_Price', 0)} to ₹{row.get('Max_Price', 0)}."
                elif collection_name == "CropProduction":
                    text = f"Crop {row.get('Crop', 'N/A')} in state {row.get('State_Name', 'N/A')}, district {row.get('District_Name', 'N/A')} during the {row.get('Season', 'N/A')} season (Year {row.get('Crop_Year', 'N/A')}) had a production of {row.get('Production', 'N/A')} units over an area of {row.get('Area', 'N/A')} hectares, resulting in a specific harvest profile."
                else:
                    text = '. '.join([f"{col}: {val}" for col, val in row.dropna().items()])
                texts.append(text)
            
            metadatas = df.to_dict(orient='records')
            
            WeaviateVectorStore.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                client=self.client,
                index_name=collection_name,
                text_key="text"
            )
            print(f"Successfully ingested {len(texts)} rows into {collection_name}")
            
        except Exception as e:
            print(f"Ingestion Error for {collection_name}: {e}")

    def get_context(self, query, collection_name, k=3):
        """Retrieves context from Weaviate V4"""
        if not self.client:
            return []
            
        try:
            vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name=collection_name,
                embedding=self.embeddings,
                text_key="text"
            )
            return vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Retrieval Error from {collection_name}: {e}")
            return []

    async def generate_recommendation(self, user_input: dict):
        """Main RAG Flow"""
        query_type = user_input.get("type", "crop_suggestion")
        
        # Build optimized search queries
        if query_type == "crop_suggestion":
            search_query = f"Crop performance in {user_input.get('districtName')}, {user_input.get('stateName')} during {user_input.get('season')}"
            context_docs = self.get_context(search_query, "CropProduction")
        else:
            search_query = f"Market prices for {user_input.get('commodity')} in {user_input.get('marketName')}"
            context_docs = self.get_context(search_query, "MandiPrices")

        context_text = "\n---\n".join([doc.page_content for doc in context_docs])
        if not context_text:
            context_text = "No direct historical matches found in dataset. Use general agronomic best practices."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Agricultural RAG Assistant. "
                       "You MUST respond with a valid JSON object. "
                       "Suggest exactly 5 RECOMMENDED CROPS. If the farmer has a 'Preferred Crop' (targetedCrop), evaluate it thoroughly and include it in the recommendations if viable. "
                       "Explain HOW the 'Preferred Crop' will be PROFITABLE to the user and its economic viability. "
                       "Focus on 'Farmer Benefit': explain why a crop is beneficial (yield, profit, or soil health). "
                       "CRITICAL: Market trends must look REALISTIC. Do not return flat or continuously increasing lines. "
                       "Include natural fluctuations (ups and downs) based on supply/demand. "
                       "STRUCTURE: "
                       "{{"
                       "  \"recommendedCrops\": [{{ \"crop\": \"Name\", \"probability\": 85, \"risk\": \"Low\" }}, ... (total 5 items)], "
                       "  \"explanation\": {{ \"bullets\": [\"Profitability of preferred crop\", \"Economic benefit for the farmer\", \"Soil compatibility logic\"] }}, "
                       "  \"pricePrediction\": {{ \"crop\": \"Top Crop\", \"rangeMin\": 2100, \"rangeMax\": 2600, \"modalPrice\": 2350, \"arrivalQuantity\": \"500-1000\", \"marketFeePct\": 1.5, \"traderCount\": 25, \"unit\": \"Quintal\", \"grade\": \"FAQ\" }}, "
                       "  \"marketTrend\": {{ \"series\": [{{ \"label\": \"Jan\", \"price\": 2100 }}, ... (6 months)] }}, "
                       "  \"risk\": {{ \"level\": \"Low\" }}"
                       "}}"),
            ("user", "Historical Context from Datasets:\n{context}\n\n"
                     "Farmer Input Details:\n{farmer_input}\n\n"
                     "Requirement: "
                     "1. Suggest 5 different crops. "
                     "2. Analyze profitability of preferred crop based on land size and input constraints. "
                     "3. Market trend must show REAL-WORLD volatility (increasing AND decreasing). "
                     "4. Explain why these crops are beneficial for the specific farmer input. "
                     "Respond with a detailed JSON analysis.")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "context": context_text,
            "farmer_input": json.dumps(user_input, indent=2)
        })
        return response.content

    async def analyze_live_market_data(self, records: list, user_input: dict):
        """Analyzes real-time Mandi records boosted by historical RAG context"""
        is_empty = len(records) == 0
        commodity = user_input.get("commodity", "this crop")
        state = user_input.get("state", "the region")
        district = user_input.get("district", "the district")
        
        # 1. Retrieve Historical Context for grounding
        hist_query = f"Market prices for {commodity} in {district}, {state}"
        hist_docs = self.get_context(hist_query, "MandiPrices", k=5)
        hist_context = "\n---\n".join([doc.page_content for doc in hist_docs])
        if not hist_context:
            hist_context = "No historical records in local dataset."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Agricultural Market Intelligence System. "
                       "You analyze live Mandi records and compare them with historical dataset context. "
                       "You MUST respond with a valid JSON object. "
                       "CRITICAL: Market trends must look REALISTIC. Do not return flat lines. "
                       "Include natural fluctuations (ups and downs) based on seasonal supply/demand logic. "
                       "STRUCTURE: "
                       "{{"
                       "  \"analysis_bullets\": [\"Dynamic analysis comparing live vs history\", \"Regional price driver insight\"], "
                       "  \"sentiment\": \"Stable/Bearish/Bullish\", "
                       "  \"prediction_insight\": \"Short prediction for next 15 days\", "
                       "  \"trend_series\": [{{ \"label\": \"Month\", \"price\": 2100 }}, ... (6 months)], "
                       "  \"modeled_modal_price\": 2350, "
                       "  \"modeled_range_min\": 2100, "
                       "  \"modeled_range_max\": 2600, "
                       "  \"fallback_suggestions\": [\"Try X market nearby\", \"Check Y variety\"], "
                       "  \"is_modeled\": true "
                       "}}"),
            ("user", "Live Records Found: {records}\n"
                     "Historical Dataset Context:\n{hist_context}\n\n"
                     "User Input: {user_input}\n"
                     "Requirement: "
                     "1. Generate a 6-month historical trend series for {commodity} in {state}. "
                     "2. IMPORTANT: The trend must show REAL-WORLD volatility (not just a straight line). "
                     "3. Compare live records (if any) with historical benchmarks from context. "
                     "4. If records are empty, simulate realistic prices based on history.")
        ])


        chain = prompt | self.llm
        response = await chain.ainvoke({
            "records": json.dumps(records, indent=2),
            "hist_context": hist_context,
            "user_input": json.dumps(user_input, indent=2),
            "commodity": user_input.get("commodity", "this crop"),
            "state": user_input.get("state", "the region")
        })
        return response.content

    async def normalize_market_params(self, raw_input: dict):
        """Uses LLM to normalize market search parameters for Mandi API"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at Indian Agricultural Market data normalization. "
                       "Convert the user's raw input into official Mandi API standard names. "
                       "Example: 'up' -> 'Uttar Pradesh', 'wheat' -> 'Wheat', 'Achnera' -> 'Achnera'. "
                       "You MUST respond with a valid JSON object. "
                       "STRUCTURE: "
                       "{{"
                       "  \"state\": \"Full State Name\", "
                       "  \"district\": \"Precise District Name\", "
                       "  \"market\": \"Specific APMC/Market Name\", "
                       "  \"commodity\": \"Official Commodity Name\" "
                       "}}"),
            ("user", "Raw Input: {raw_input}")
        ])

        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"raw_input": json.dumps(raw_input)})
            clean_json = response.content.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            return json.loads(clean_json)
        except:
            return raw_input # Fallback to raw if LLM fails

    async def chat_with_context(self, message: str, history: list = []):
        """High-fidelity chat with RAG context and session history"""
        # 1. Expand context retrieval
        docs = self.get_context(message, "CropProduction", k=5)
        docs += self.get_context(message, "MandiPrices", k=5)
        
        context_text = "\n---\n".join([doc.page_content for doc in docs])
        
        # 2. Refined Prompt for Factuality and Proper Formatting
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are 'Crop AI Expert', an exclusive Agricultural Advisory Assistant for farmers. "
                       "Your goal is to provide HIGHLY FACTUAL and 100% SUPPORTIVE advice. "
                       "STRICT RULES: "
                       "1. ONLY answer questions related to farming, agriculture, crops, weather, pesticides, fertilizers, capital, and markets. "
                       "   If a user asks about non-farming topics, strictly refuse and state you are an Agricultural Expert. "
                       "2. Suggest best crops by analyzing the provided datasets. "
                       "3. Suggest best fertilizers, pesticides, farming-related chemicals, and capital/financial strategies when asked. "
                       "4. NEVER say things like 'I don't have access to the real world' or 'I am an AI'. If you lack data, act as if you are extracting the most up-to-date best practices from the internet using your vast general knowledge. "
                       "5. Provide a 'reply' and a 'suggested' list of 3 specific follow-up questions. "
                       "6. FORMATTING: In the 'reply' field, use a combination of short paragraphs and bullet points (using \n and -) to make the text easy to read for the farmer. "
                       "7. CRITICAL: Output ONLY the raw JSON object. Do NOT wrap it in markdown code blocks like ```json or use any other surrounding text. "
                       "8. Respond in a clean JSON format."),
            ("user", "FARMER DATA CONTEXT:\n{context}\n\n"
                     "CONVERSATION HISTORY (Last 5 interactions):\n{history}\n\n"
                     "CURRENT FARMER QUESTION: {message}")
        ])

        chain = prompt | self.llm
        
        # Format history for the prompt (mimicking ConversationBuffer)
        formatted_history = ""
        for h in history[-5:]:
            role = "Farmer" if h.get("role") == "user" else "Advisor"
            content = h.get("content") or h.get("text")
            formatted_history += f"{role}: {content}\n"

        response = await chain.ainvoke({
            "context": context_text if context_text else "No historical records found for this specific query.",
            "history": formatted_history if formatted_history else "Initial Conversation",
            "message": message
        })
        
        import re
        
        # Clean up JSON using regex to find the first { and last }
        content = response.content.strip()
        try:
            # Try to extract the first valid-looking JSON object
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                clean_json = match.group(1)
                return json.loads(clean_json)
            else:
                raise ValueError("No JSON block found")
        except Exception as e:
            print(f"JSON Parsing Error: {e} | Raw: {content[:100]}...")
            # Emergency Fallback: If it fails, assume the content IS the reply
            # and strip away the outside JSON structure if it leaked through
            display_text = content
            if display_text.startswith('{') and '"reply"' in display_text:
                try:
                    # Last ditch effort to manual extract reply
                    reply_match = re.search(r'"reply":\s*"(.*?)"', display_text, re.DOTALL)
                    if reply_match:
                        display_text = reply_match.group(1).replace('\\n', '\n')
                except:
                    pass
            
            return {
                "reply": display_text, 
                "suggested": ["How to optimize yield?", "Current price trends?", "Pest control tips?"]
            }

    async def generate_dynamic_heatmap(self, state: str, district: str) -> list:
        """Uses the LLM's vast general knowledge to generate a highly realistic Heat Map for all mandis in a district."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Indian Agricultural Geographer. You have access to real-world internet knowledge."
                       "You must return ONLY a JSON array. Do not use backticks or markdown blocks."
                       "Given a State and District, list ALL the ACTUAL, REAL-WORLD local Mandis/APMCs (e.g., use 'Lasalgaon' instead of 'Niphad', or 'Vashi APMC' instead of 'Mumbai'). "
                       "For each Mandi, list 3 to 5 crops realistically traded there. "
                       "Assign an 'intensity' number (raw numerical volume metric for charting). "
                       "Also assign 'production_label' (e.g., '14,000 MT' or '25,000 Quintals'). "
                       "And include a short 'analysis_text' explaining its regional dominance. "
                       "STRUCTURE: "
                       "[{{\"subRegion\": \"Actual Mandi Name\", \"cropName\": \"Crop A\", \"intensity\": 15000, \"production_label\": \"15,000 Quintals\", \"analysis_text\": \"Dominant onion market...\"}}, ...]"),
            ("user", f"Provide the crop intensity heatmap data for all mandis in {district}, {state}.")
        ])

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({})
            # Clean JSON
            clean_json = response.content.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            # Simple bracket extraction for safety
            import re
            match = re.search(r'(\[.*\])', clean_json, re.DOTALL)
            if match:
                clean_json = match.group(1)
            
            return json.loads(clean_json)
        except Exception as e:
            print("Dynamic heatmap generation failed:", e)
            return []

    def __del__(self):
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except:
                pass

# Global singleton (will be initialized on first use)
_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service