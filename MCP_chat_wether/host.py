
import os
import asyncio
from google import genai
from google.genai import types

class WeatherClient:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            return

        self.client = genai.Client(api_key=api_key)
        self.model_id = self._find_best_model()

    def _find_best_model(self):
        """Scans models and selects the best one"""
        try:
            available_models = list(self.client.models.list())
            for m in available_models:
                if "flash" in m.name.lower():
                    return m.name
            if available_models:
                return available_models[0].name
            return "gemini-1.5-flash"
        except Exception as e:
            print(f"Error scanning models: {e}")
            return "gemini-1.5-flash"

    async def process_query(self, query_text):
        if not self.model_id:
            return

        try:
            # הגדרת הכלים והקונפיגורציה
            tools = [types.Tool(google_search=types.GoogleSearch())]
            
            # עדכון הנחיית המערכת: תמיד לענות בעברית
            config = types.GenerateContentConfig(
                system_instruction="You are a helpful assistant providing weather information. Always respond in Hebrew.",
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
            )

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=query_text,
                config=config
            )

            if response.text:
                print("\n" + "="*30)
                print(f"Answer: {response.text}")
                print("="*30 + "\n")
            else:
                print("\n[Model is processing or calling tools...]\n")
            
            return response

        except Exception as e:
            print(f"Processing error: {e}")

# פונקציית ההרצה האינטראקטיבית
async def main():
    client = WeatherClient()
    if not hasattr(client, 'client'):
        return

    print("Welcome to the Weather Chat!")
    print("Enter a city name (or 'exit' to quit):")

    while True:
        # קבלת קלט מהמשתמש באנגלית
        user_input = input("City to check > ")
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'יציאה']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # השאילתה נשלחת למודל עם הקשר ברור
        query = f"What is the current weather in {user_input}?"
        await client.process_query(query)

if __name__ == "__main__":
    asyncio.run(main())