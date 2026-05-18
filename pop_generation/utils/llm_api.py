import openai
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
openai.api_base = "https://api.chatanywhere.tech/v1"
openai.api_key = "sk-REDACTED"  # remove real key when publishing

def get_res_from_chat(messages, args):
    max_tokens = 256 if args.type == 'paraphrase' else 16
    while True:
        try:
            res = openai.ChatCompletion.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=max_tokens,
                logprobs=True
            )
            return res['choices'][0]['message']['content'].strip()
        except openai.error.RateLimitError as e:
            print('\nRateLimitError\t', e, '\tRetrying...')
            time.sleep(5)
        except openai.error.ServiceUnavailableError as e:
            print('\nServiceUnavailableError\t', e, '\tRetrying...')
            time.sleep(5)
        except openai.error.Timeout as e:
            print('\nTimeout\t', e, '\tRetrying...')
            time.sleep(5)
        except openai.error.APIError as e:
            print('\nAPIError\t', e, '\tRetrying...')
            time.sleep(5)
        except openai.error.APIConnectionError as e:
            print('\nAPIConnectionError\t', e, '\tRetrying...')
            time.sleep(5)
        except Exception as e:
            print(e)
            return None


def get_llm_result(prompts, samples, args):
    results = [None] * len(prompts)

    def process_prompt(index, prompt, sample):
        messages = [{"role": "user", "content": prompt}]
        response = get_res_from_chat(messages, args)
        return index, {
            'qa_prompt': prompt,
            'Res': response
        }

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        future_to_index = {
            executor.submit(process_prompt, i, prompts[i], samples[i]): i
            for i in range(len(prompts))
        }

        for future in as_completed(future_to_index):
            try:
                index, result = future.result()
                res_text = result.get('Res') if isinstance(result, dict) else result
                pop_value = None
                try:
                    if isinstance(res_text, str):
                        import re
                        m = re.search(r"(-?\d+)", res_text)
                        if m:
                            val = int(m.group(1))
                            if val < 1:
                                val = 1
                            if val > 10:
                                val = 10
                            pop_value = val
                except Exception:
                    pop_value = None

                if isinstance(result, dict):
                    result['pop_value'] = pop_value
                else:
                    result = {'qa_prompt': prompts[index], 'Res': res_text, 'pop_value': pop_value}

                results[index] = result
            except Exception as e:
                print(f"Error processing prompt at index {future_to_index[future]}: {e}")

    return results
