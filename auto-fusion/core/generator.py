"""
Filename: generator.py
Description: LLM 接口层，负责发送 Prompt 并解析生成的 Fusion 架构代码。
             支持 Mock 模式和 Google Gemini API。
Author: Auto-Fusion Assistant
"""

import os
import re
import logging
import textwrap
from typing import Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Setup Logging
logger = logging.getLogger(__name__)

class AutoFusionGenerator:
    """
    连接 LLM 与 Auto-Fusion 系统的生成器。
    职责:
    1. 发送 Prompt 到 LLM (Gemini API).
    2. 解析返回文本，提取 'Thought' 和 'Code'。
    3. 提供 Mock 模式用于系统测试。
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None, mock_mode: bool = False):
        self.mock_mode = mock_mode
        # Use gemini-2.0-flash as a fallback or default if 3-flash-preview is unstable
        # But user requested 3-flash-preview. Let's keep it configurable.
        # The error was "NotFound", meaning the model name might be invalid for this API key or region.
        # "gemini-3-flash-preview" might not be publicly available yet or requires specific access.
        # Fallback logic: try the requested model, if it fails, fallback to a known stable model like gemini-1.5-flash.
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.mock_mode:
            if not self.api_key:
                raise ValueError("Missing GOOGLE_API_KEY for real generation.")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini Generator with model: {self.model_name}")
        else:
            logger.info("Initialized Generator in Mock Mode.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _call_real_api(self, prompt: str) -> Tuple[str, any]:
        """
        Calls Google Gemini API with retries and permissive safety settings.
        Handles model fallback if NotFound error occurs.
        """
        # Safety Settings: Block NONE to allow code generation
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.9, # High creativity for NAS
            candidate_count=1,
            max_output_tokens=2048, # Ensure enough space for code
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            # Check for valid response
            if not response.parts:
                logger.warning(f"Empty response from Gemini. Feedback: {response.prompt_feedback}")
                raise ValueError("Empty response from API")
                
            return response.text, response.usage_metadata
            
        except Exception as e:
            # Handle NotFound (404) specifically - usually wrong model name
            if "404" in str(e) or "NotFound" in str(e):
                logger.warning(f"Model '{self.model_name}' not found or not accessible. Trying fallback 'gemini-2.0-flash'...")
                try:
                    self.model_name = "gemini-2.0-flash"
                    self.model = genai.GenerativeModel(self.model_name)
                    # Retry immediately with new model
                    return self._call_real_api(prompt)
                except Exception as fallback_e:
                    logger.error(f"Fallback model failed too: {fallback_e}")
                    raise fallback_e
            
            logger.error(f"Gemini API Error: {e}")
            raise

    def _parse_output(self, raw_text: str) -> Dict[str, str]:
        """
        解析 LLM 的原始输出，提取 Thought 和 Code Block。
        """
        result = {'thought': '', 'code': ''}
        
        # 1. Extract Code Block (```python ... ``` or just ``` ... ```)
        # Use DOTALL to match newlines inside the block
        code_match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
        
        if code_match:
            result['code'] = code_match.group(1).strip()
            
            # 2. Extract Thought (Everything before the code block)
            # Or look for explicit "Thought:" marker
            pre_code_text = raw_text[:code_match.start()].strip()
            
            # Try to find "Thought:" or "Reasoning:" explicitly
            thought_match = re.search(r"(?:Thought|Reasoning):\s*(.*)", pre_code_text, re.DOTALL | re.IGNORECASE)
            if thought_match:
                result['thought'] = thought_match.group(1).strip()
            else:
                # If no marker, assume everything before code is thought
                result['thought'] = pre_code_text
        else:
            # Fallback: If no code blocks, maybe the whole text is code?
            # Or just warn and return empty code
            logger.warning("No code block found in LLM response.")
            result['thought'] = raw_text # Treat all as text
            
        return result

    def generate(self, prompt: str, mock_mode: Optional[bool] = None) -> Dict[str, str]:
        """
        生成融合架构代码。
        
        Args:
            prompt: 完整的提示词。
            mock_mode: Override initialization mock_mode if provided.
            
        Returns:
            {'thought': str, 'code': str}
        """
        # Determine effective mode
        is_mock = self.mock_mode if mock_mode is None else mock_mode
        
        if is_mock:
            logger.info("Generator running in Mock Mode.")
            return self._mock_generation()
            
        logger.info(f"Sending prompt to Gemini ({self.model_name})...")
        try:
            raw_text, usage = self._call_real_api(prompt)
            logger.info(f"Received response. Usage: {usage}")
            return self._parse_output(raw_text)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback to mock if API fails? Or re-raise?
            # For NAS, failing hard is better than silent mock fallback during real runs.
            raise e
