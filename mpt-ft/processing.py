def preprocess_rationale_ft(inp:dict, prompt_preface:str):
    return {'prompt': prompt_preface+inp['query'], 'response': inp['rationale']}