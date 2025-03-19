# chat.py
import random
import json
import torch
from collections import deque
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words, context_vector, prepare_full_input

class ChatContextManager:
    def __init__(self, max_history=3):
        self.active_contexts = set()
        self.context_history = deque(maxlen=max_history)
        self.pending_requirements = set()  # Stores tuples

    def update_context(self, new_contexts):
        self.context_history.append(self.active_contexts.copy())
        self.active_contexts = set(new_contexts)
        self.pending_requirements.clear()

    def add_requirement(self, requirements):
        if requirements:
            # Convert list to tuple for hashability
            req_tuple = tuple(requirements) if isinstance(requirements, list) else requirements
            self.pending_requirements.add(req_tuple)

    def validate_requirements(self):
        return all(
            req in self.active_contexts
            for req_tuple in self.pending_requirements
            for req in req_tuple
        )

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and data
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

model_data = torch.load('data.pth', map_location=device)
model = NeuralNet(
    input_size=model_data['input_size'],
    hidden_size=model_data['hidden_size'],
    num_classes=model_data['output_size'],
    context_size=len(model_data.get('context_tags', []))
).to(device)
model.load_state_dict(model_data['model_state'])
model.eval()

# Initialize context system
context_manager = ChatContextManager()
all_context_tags = model_data.get('context_tags', [])
BOW_SIZE = len(model_data['all_words'])

bot_name = "KOSISOCHUKWUBOT"
print(f"{bot_name}: Ready for interaction! Type 'reset' to clear context or 'quit' to exit.")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'reset':
        context_manager = ChatContextManager()
        print(f"{bot_name}: Context reset complete")
        continue

    # Preprocess input
    tokens = tokenize(user_input)
    bow = bag_of_words(tokens, model_data['all_words'])
    ctx_vec = context_vector(context_manager.active_contexts, all_context_tags)
    full_input = torch.from_numpy(prepare_full_input(bow, ctx_vec)).float().unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(full_input)
    
    prob = torch.softmax(output, dim=1)
    top_prob, top_idx = torch.max(prob, dim=1)
    predicted_tag = model_data['tags'][top_idx.item()]

    if top_prob.item() > 0.48:
        response_given = False
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                # Handle context requirements
                context_reqs = intent.get('context_required', [])
                if context_reqs:
                    context_manager.add_requirement(tuple(context_reqs))  # Convert to tuple
                
                if not context_manager.validate_requirements():
                    missing = [req for req in context_reqs if req not in context_manager.active_contexts]
                    print(f"{bot_name}: Need context: {', '.join(missing)}. Please provide relevant information first.")
                    response_given = True
                    break

                # Update context state
                if 'context_set' in intent:
                    context_manager.update_context(intent['context_set'])
                
                # Select response
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                response_given = True
                break
        
        if not response_given:
            print(f"{bot_name}: I'm not sure how to respond to that in the current context")
    else:
        print(f"{bot_name}: Could you rephrase or provide more context information?")

    # Auto-context cleanup after 5 inactive turns
    if len(context_manager.context_history) >= 5 and not context_manager.active_contexts:
        context_manager.update_context([])
        print(f"{bot_name}: [System] Conversation context automatically refreshed")