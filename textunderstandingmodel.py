#text understanding model

import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch 
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE


#load pre-trained BERT and tokenizer 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#sample for sentiment analysis 
train_data = {
    "query": [
        "What should I wear for a casual outing?",
        "I need a formal outfit for a job interview.",
        "I want to dress up for a party tonight.",
        "Do you have any suggestions for a sporty attire?",
        "I'm going to a wedding, what should I wear?",
        "My friends invited me to a small gathering, give me a chill outfit.",
        "What should I wear to my graduation dinner?",
        "I need an outfit for my birthday party.",
        "I need an outfit for a wedding",
        "I need a formal outfit for a job interview",
        "Style me for my graduation dinner",
        "I'm going to a wedding, what should I wear?",
        "what should I wear for a casual date?",
        "What should I wear for a casual gathering with coworkers?",
        "What would be suitable for a relaxed weekend?",
        "I'm casually chilling with friends, what should I wear",
        "Going to a friend's birthday party later, please make me an oufit",
        "Make me an outfit for my best friend's birthday party",
        "Dress me for a night out at a club",
        "I want to dress up for a party tonight",
        "I'm attending a cocktail party this weekend, what should I wear?",
        "My friend's birthday is coming up, suggest me an outfit!",
        "I need an outfit for a themed costume party next week.",
        "What's a suitable outfit for a gala party?",
        "Going to a bar/ club tonight, give me an outfit to wear.",
        "I'm invited to a rooftop party, what's the appropriate attire?",
        "Attending a christmas party, need suggestions for a good outfit.",
        "I have a corporate dinner with clients, what's the formal dress code?",
        "I'm presenting at a conference, need a formal look.",
        "Formal event at the embassy, what should I wear?",
        "Attending a formal ceremony, need a outfit.",
        "I'm a keynote speaker at a symposium, suggest me a formal outfit.",
        "What's the appropriate attire for a formal reception?",
        "Formal charity fundraiser next month, need outfit ideas.",
        "I'm planning a weekend brunch with friends, what's a casual-chic outfit?",
        "What's a comfortable outfit for a casual day of errands?",
        "Casual movie night with friends, need outfit inspiration.",
        "I'm volunteering at the animal shelter, suggest me a casual outfit.",
        "Going for a casual stroll in the park, what should I wear?",
        "Relaxing at home with family, need casual loungewear.",
    ],

    "label": [
        "casual",
        "formal",
        "party",
        "casual",
        "formal",
        "casual",
        "formal",
        "party",
        "formal",
        "formal",
        "formal",
        "formal",
        "casual",
        "casual",
        "casual",
        "casual",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "casual",
        "casual",
        "casual",
        "casual",
        "casual",
        "casual",
    ]
}

eval_data = {
    "query": [
        "What should I wear for a casual gathering with coworkers?",
        "I'm looking for something professional to wear to a meeting.",
        "What would be suitable for a relaxed weekend?",
        "I need something stylish for a special occasion.",
        "Can you suggest something comfortable for outdoor activities?",
        "Give me a fun outfit to wear to a baby shower",
        "What's a good outfit for my anniversary date?",
        "Going out with friends to a club, style me",
        "Give me a formal outfit to wear to an office meeting",
        "style me for a formal reception",
        "What to wear to a formal dinner at a fancy restaurant?",
        "Please give me outfit ideas for a wedding",
        "Chilling with friends, what to wear?",
        "Casual dinner later, style me",
        "What should I wear for a casual night in with close friends?",
        "Could you give me some ideas on what to wear for a casual lunch with coworkers?",
        "Going clubbing later, what should I wear?",
        "Style me for a frat party later!",
        "There's a birthday party later, give me some outfit ideas!",
        "what should I wear to a party?",
        "I'm hosting a dinner party at my place, what's a stylish outfit?",
        "Attending a friend's housewarming party, need suggestions for a look.",
        "What's a suitable outfit for a summer pool party?",
        "I'm going to a themed costume party, suggest me a fun and creative ensemble.",
        "Celebrating New Year's Eve at a rooftop bar, need outfit ideas.",
        "Attending a holiday masquerade party, what's the appropriate attire?",
        "I'm invited to a friend's engagement party, suggest me an outfit for the occasion.",
        "I have a job interview, need a formal outfit.",
        "Attending a formal fundraiser, what's the dress code?",
        "Formal dinner with colleagues and clients, need a sophisticated yet modern look.",
        "Please suggest me a something formal to wear.",
        "Attending a formal reception, what should I wear?",
        "Formal dinner at a restaurant, what should I wear?",
        "Formal networking event next week, need suggestions for an outfit.",
        "Planning a cozy night in with my significant other, what's a casual outfit?",
        "Casual picnic in the park with friends, suggest me a laid-back outfit.",
        "I'm going for a casual coffee date, need outfit ideas.",
        "Attending a casual outdoor concert, what should I wear?",
        "Casually exploring, need suggestions for an outfit.",
        "Casual weekend brunch with family, what's a casual-chic outfit for a brunch gathering?",

    ],

    "label": [
        "casual",
        "formal",
        "casual",
        "party",
        "casual",
        "party",
        "formal",
        "party",
        "formal",
        "formal",
        "formal",
        "formal",
        "casual",
        "casual",
        "casual",
        "casual",
        "party",
        "party",
        "party",
        "party",
         "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "party",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "formal",
        "casual",
        "casual",
        "casual",
        "casual",
        "casual",
        "casual",
    ]
}

train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)

print(train_dataset[0])
print(eval_dataset[2])

#tokenize query
def tokenize_function(example):
    return tokenizer(example["query"], padding="max_length", truncation=True, return_tensors="pt")

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

#label to numerical indices
unique_labels_train = set(train_dataset["label"])
label_map_train = {label: idx for idx, label in enumerate(unique_labels_train)}

def map_labels_train(batch):
    batch["label"] = torch.tensor([label_map_train[label] for label in batch["label"]], dtype=torch.long)
    return batch

train_dataset = train_dataset.map(map_labels_train, batched=True)

unique_labels_eval = set(eval_dataset["label"])
label_map_eval = {label: idx for idx, label in enumerate(unique_labels_eval)}

def map_labels_eval(batch):
    batch["label"] = torch.tensor([label_map_eval[label] for label in batch["label"]], dtype=torch.long)
    return batch

eval_dataset = eval_dataset.map(map_labels_eval, batched=True)

#format to pytorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

num_classes = len(set(train_dataset["label"]))
model.config.num_labels = num_classes

model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# Set up data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Set up loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set up number of epochs
num_epochs = 3

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        #forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        #backward pass/ optimization
        loss.backward()
        optimizer.step()
        train_loss +=loss.item()*input_ids.size(0)
        
    train_loss/=len(train_loader.dataset)

    # Evaluation loop
    model.eval()
    eval_loss = 0
    num_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
    
            #forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            eval_loss += loss.item()*input_ids.size(0)

            _, predicted_labels = torch.max(logits, dim=1)
            num_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    eval_loss /= len(eval_loader.dataset)
    accuracy = num_correct / total_samples

    print(f"Epoch {epoch + 1}:")
    print(f"  Training Loss: {train_loss:.4f}")
    print(f"  Evaluation Loss: {eval_loss:.4f}")
    print(f"  Evaluation Accuracy: {accuracy:.4f}")

#torch.save(model, 'fashionai/textunderstandingmodel.pth')

model.save_pretrained("fashionai/textunderstandingmodel")

textunderstandingmodel = BertForSequenceClassification.from_pretrained("fashionai/textunderstandingmodel")


# Tokenize user input
user_input = [
    "I need an outfit for a formal wedding",
    "I need a formal outfit for a job interview",
    "Style me for my graduation dinner",
    "I'm going to a formal reception, what should I wear?",
    "Give me a formal outfit to wear to an office meeting",
    "style me for a formal reception",
    "What to wear to a formal dinner at a fancy restaurant?",
    "Please give me outfit ideas for a wedding",
    "what should I wear for a casual date?",
    "Casual movie night with friends, need outfit inspiration.",
    "I'm volunteering at the animal shelter, suggest me a casual outfit.",
    "Going for a casual stroll in the park, what should I wear?",
    "Relaxing at home with family, need casual loungewear.",
    "What should I wear for a casual gathering with coworkers?",
    "What would be suitable for a relaxed weekend?",
    "I'm casually chilling with friends, what should I wear",
    "Going to a friend's birthday party later, please make me an oufit",
    "Make me an outfit for my best friend's birthday party",
    "Dress me for a night out at a club",
    "I want to dress up for a party tonighht",
    "Going to a friend's birthday party later, please make me an oufit",
    "Make me an outfit for my best friend's birthday party",
    "Dress me for a night out at a club",
    "I want to dress up for a party tonight"
]

formal_user_input = [
    "I need an outfit for a formal wedding",
    "I need a formal outfit for a job interview",
    "Style me for my graduation dinner",
    "I'm going to a formal reception, what should I wear?",
    "Give me a formal outfit to wear to an office meeting",
    "style me for a formal reception",
    "What to wear to a formal dinner at a fancy restaurant?",
    "Please give me outfit ideas for a wedding",
    "I need an outfit for a formal business conference.",
    "What should I wear for a formal event?",
    "Attending a formal corporate dinner, need outfit ideas.",
    "I'm a keynote speaker at a conference, suggest me a formal attire.",
    "Dress me for a formal  fundraiser.",
    "What's appropriate attire for a formal  reception?",
    "I'm attending a formal awards ceremony, what should I wear?",
    "Going to a formal dinner, need outfit inspiration.",
    "I need a formal outfit for an event.",
    "Attending a formal performance, suggest me an outfit.",
    "What should I wear for a formal evening wedding?",
    "Dress me for a formal party.",
    "I'm presenting at a formal conference, need an outfit.",
    "Attending a formal art gallery opening, what's suitable attire?",
    "I need a formal outfit for a theater premiere.",
    "What should I wear for a formal holiday lunch?",
    "Attending a wedding, need outfit suggestions.",
    "What should I wear to a formal charity gala?",
    "I need a formal outfit for a business conference.",
    "Give me an formal outfit for a  reception.",
    "Style me for a formal awards ceremony.",
    "I'm attending a formal performance, what's appropriate attire?",
    "Dress me for a formal dinner with international delegates.",
    "Suggest me an outfit for a formal art gallery opening.",
    "I need a formal outfit for a corporate event.",
    "What should I wear for a formal  fundraiser?",
    "Make me an outfit for a formal business luncheon.",
    "Attending a formal concert, need attire advice.",
    "I want to dress to impress at a formal lunch.",
    "What's the appropriate dress code for a formal luncheon?",
    "Give me an outfit for a formal reception at a luxury hotel."
    "I need a polished outfit for a formal reception."
]

casual_user_input = [
    "what should I wear for a casual date?",
    "Casual movie night with friends, need outfit inspiration.",
    "I'm volunteering at the animal shelter, suggest me a casual outfit.",
    "Going for a casual stroll in the park, what should I wear?",
    "Relaxing at home with family, need casual loungewear.",
    "What should I wear for a casual gathering with coworkers?",
    "What would be suitable for a relaxed weekend?",
    "I'm casually chilling with friends, what should I wear",
    "What should I wear for a casual day out with friends?",
    "Going for a casual lunch date, suggest me an outfit.",
    "Casual movie night at home, need comfy outfit ideas.",
    "What's suitable attire for a casual family gathering?",
    "I'm volunteering at the local food bank, need a casual outfit.",
    "Going for a casual hike this weekend, what should I wear?",
    "Relaxing at home with a book, need cozy casual wear.",
    "Casual brunch, what's appropriate attire?",
    "I'm attending a casual barbecue, suggest me an outfit.",
    "Going for a casual bike ride in the park, need outfit ideas.",
    "What should I wear for a casual dinner at a friend's house?",
    "Casual coffee meetup with an old friend, need outfit inspiration.",
    "I'm taking a casual stroll along the beach, what should I wear?",
    "Going for a casual shopping trip, need comfortable attire.",
    "Relaxing weekend at home, need casual loungewear.",
    "Casual day at the office, what's suitable for a relaxed dress code?",
    "Planning a casual brunch with friends, what should I wear?",
    "I'm going for a casual picnic in the park, need outfit ideas.",
    "Suggest me an outfit for a casual weekend getaway.",
    "Going for a casual bike ride, what's suitable attire?",
    "What should I wear for a casual movie night at home?",
    "I need an outfit for a casual family gathering.",
    "Going for a casual walk along the beach, what to wear?",
    "I want to dress casually for a day of shopping, any suggestions?",
    "Suggest me a relaxed outfit for a casual dinner date.",
    "Attending a casual outdoor concert, need attire advice.",
    "I'm hosting a casual game night, what's appropriate attire?",
    "Going for a casual coffee date, what should I wear?",
    "I need an outfit for a casual weekend barbecue.",
    "What's suitable attire for a casual day of sightseeing?",
    "Give me an outfit for a casual hangout with friends.",
    "Planning a casual evening stroll, need outfit inspiration."
]

party_user_input = [
    "Going to a friend's birthday party later, please make me an oufit",
    "Make me an outfit for my best friend's birthday party",
    "Dress me for a night out at a club",
    "I want to dress up for a party tonighht",
    "Going to a friend's birthday party later, please make me an oufit",
    "Make me an outfit for my best friend's birthday party",
    "Dress me for a night out at a club",
    "I want to dress up for a party tonight",
    "I'm attending a friend's birthday party, suggest me an outfit.",
    "Going to a fancy cocktail party, what should I wear?",
    "Dress me for a night out at a club with friends.",
    "I want to dress up for a party tonight, any outfit ideas?",
    "Make me an outfit for my best friend's birthday party.",
    "What should I wear for a holiday  party?",
    "Attending a party, need outfit inspiration.",
    "Suggest me an outfit for a New Year's party.",
    "Going to a costume party, what should I wear?",
    "I'm hosting a dinner party, need an outfit.",
    "Dress me for a glamorous red carpet event.",
    "Attending a friend's party, need outfit ideas.",
    "What should I wear for a party?",
    "I want to go to a party, suggest me an outfit.",
    "Going to a party, what's appropriate attire?",
    "Make me an outfit for a party.",
    "I need a standout outfit for a cocktail party.",
    "What should I wear to a chic rooftop party?",
    "Give me an outfit for a nightclub event.",
    "I'm attending a fancy dress costume party, need attire suggestions.",
    "Style me for a festive holiday office party.",
    "Suggest me an outfit for a lively housewarming party.",
    "Going to a New Year's Eve party, what should I wear?",
    "Dress me for a beachside bonfire party.",
    "I want to stand out at a party, what's the best attire?",
    "Attending a themed party, need outfit inspiration.",
    "I'm going to a poolside summer party, what's appropriate attire?",
    "Give me an outfit for a wild bachelorette party.",
    "What should I wear for a glamorous party?",
    "Suggest me an outfit for a fun-filled birthday party.",
    "I need an eye-catching outfit for a VIP club party.",
    "Style me for a sophisticated garden party."
]

embeddings = []
formal_embeddings = []
casual_embeddings = []
party_embeddings = []

# Process each input text
for text in user_input:
    # Tokenize input text
    tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Run inference on the user input to obtain embeddings
    with torch.no_grad():
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        output = textunderstandingmodel(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]  # Assuming the last layer's hidden states are at index 0
        embeddings.append(last_hidden_state)

embeddings_tensor = torch.stack(embeddings)

# Process each input text
for text in formal_user_input:
    # Tokenize input text
    tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Run inference on the user input to obtain embeddings
    with torch.no_grad():
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        output = textunderstandingmodel(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]  # Assuming the last layer's hidden states are at index 0
        formal_embeddings.append(last_hidden_state)

formal_embeddings_tensor = torch.stack(formal_embeddings)

for text in casual_user_input:
    # Tokenize input text
    tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Run inference on the user input to obtain embeddings
    with torch.no_grad():
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        output = textunderstandingmodel(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]  # Assuming the last layer's hidden states are at index 0
        casual_embeddings.append(last_hidden_state)

casual_embeddings_tensor = torch.stack(casual_embeddings)

for text in party_user_input:
    # Tokenize input text
    tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Run inference on the user input to obtain embeddings
    with torch.no_grad():
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        output = textunderstandingmodel(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]  # Assuming the last layer's hidden states are at index 0
        party_embeddings.append(last_hidden_state)

party_embeddings_tensor = torch.stack(party_embeddings)



print(embeddings_tensor.shape)  # Shape should be (num_inputs, embedding_dim)




# Save the embeddings
#torch.save(user_embeddings, "user_embeddings.pt")

#visualize embeddings

# Reduce dimensionality of embeddings using t-SNE
tsne = TSNE(n_components=2, perplexity=5, learning_rate=100)
embeddings_tsne = tsne.fit_transform(embeddings_tensor.squeeze().numpy())
formal_embeddings_tsne = tsne.fit_transform(formal_embeddings_tensor.squeeze().numpy())
casual_embeddings_tsne = tsne.fit_transform(casual_embeddings_tensor.squeeze().numpy())
party_embeddings_tsne = tsne.fit_transform(party_embeddings_tensor.squeeze().numpy())

# Plot the embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.5)
for i, txt in enumerate(user_input):
    plt.annotate(txt, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Visualization of Embeddings from User Query')
plt.show()

# Process each input text for each category
embeddings = {'Formal': [], 'Casual': [], 'Party': []}
user_input_lists = [formal_user_input, casual_user_input, party_user_input]

for category, user_input_list in zip(embeddings.keys(), user_input_lists):
    for text in user_input_list:
        # Tokenize input text
        tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Run inference on the user input to obtain embeddings
        with torch.no_grad():
            input_ids = tokenized_input['input_ids']
            attention_mask = tokenized_input['attention_mask']
            output = textunderstandingmodel(input_ids, attention_mask=attention_mask)
            last_hidden_state = output[0]  # Assuming the last layer's hidden states are at index 0
            embeddings[category].append(last_hidden_state)

# Stack embeddings into tensors for each category
embeddings_tensors = {category: torch.stack(embeddings_list) for category, embeddings_list in embeddings.items()}

# Reduce dimensionality of embeddings using t-SNE for each category
tsne_results = {}
for category, embeddings_tensor in embeddings_tensors.items():
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=100)
    embeddings_tsne = tsne.fit_transform(embeddings_tensor.squeeze().numpy())
    tsne_results[category] = embeddings_tsne

# Plot the embeddings
plt.figure(figsize=(10, 8))
colors = {'Formal': 'pink', 'Casual': 'purple', 'Party': 'blue'}  # Specify colors for each category
for category, embeddings_tsne in tsne_results.items():
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors[category], alpha=0.5, s=30)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Embeddings')
plt.legend()
plt.show()

#plot colors!! 
plt.figure(figsize=(10, 8))
plt.scatter(formal_embeddings_tsne[:, 0], formal_embeddings_tsne[:, 1], color='pink', alpha=0.5, s=30)
plt.scatter(casual_embeddings_tsne[:, 0], casual_embeddings_tsne[:, 1], color='purple', alpha=0.5, s=30)
plt.scatter(party_embeddings_tsne[:, 0], party_embeddings_tsne[:, 1], color='blue', alpha=0.5, s=30)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Visualization of Different Ocassion Categories')
plt.legend()
plt.show()
