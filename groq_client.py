# groq_client.py

import requests
import json

class GroqClient:
    """Client for Groq API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completions_create(self, model, messages, temperature=0.3, max_tokens=1200):
        """Create chat completion using Groq API"""
        
        url = f"{self.base_url}/chat/completions"
        
        # Ensure messages are properly formatted
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Groq only accepts 'user', 'assistant', or 'system' roles
            if role not in ["user", "assistant", "system"]:
                role = "user"
            
            formatted_messages.append({
                "role": role,
                "content": str(content)  # Ensure content is string
            })
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=60
            )
            
            # Print error details for debugging
            if response.status_code != 200:
                print(f"❌ Groq API Error: {response.status_code}")
                print(f"Response: {response.text}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    pass
            
            response.raise_for_status()
            return GroqResponse(response.json())
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            print(f"Groq API Error: {str(e)}")
            raise

class GroqResponse:
    """Wrapper for Groq API response"""
    
    def __init__(self, response_data):
        self.choices = [GroqChoice(choice) for choice in response_data.get("choices", [])]
        self.usage = response_data.get("usage", {})
        self.model = response_data.get("model", "")

class GroqChoice:
    """Wrapper for choice object"""
    
    def __init__(self, choice_data):
        message_data = choice_data.get("message", {})
        self.message = GroqMessage(message_data)
        self.finish_reason = choice_data.get("finish_reason", "")
        self.index = choice_data.get("index", 0)

class GroqMessage:
    """Wrapper for message object"""
    
    def __init__(self, message_data):
        self.content = message_data.get("content", "")
        self.role = message_data.get("role", "assistant")
