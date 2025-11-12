from tqdm import tqdm
import torch
from transformers import AutoProcessor
from syn_doc.scripts.tools.metrics import calculate_anls
loss_function = torch.nn.CrossEntropyLoss()
from torch.nn import CrossEntropyLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def train(model,train_dataloader,lr):
    model.train()
    predict_text_list = []
    target_text_list = []
    anls_scores = []
    predict_entity_list = []
    target_id_list = []
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr) # change learning rate

    for _, data in tqdm(enumerate(train_dataloader, 0)):
        # Convert tensors to the correct types
        input_ids = data['input_ids'].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data['attention_mask'].to(device, dtype=torch.float)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        bbox = data['bbox'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        encoded_token_objt_ids = data['token_objt_ids'].to(device, dtype=torch.long) ## For token aggregate into entities

        visual_feat = data['visual_feat'].to(device, dtype = torch.float)
        bert_cls = data['bert_cls'].to(device, dtype = torch.float)
        positional_encoding = data['positional_encoding'].to(device, dtype = torch.float)
        norm_bbox = data['norm_bbox'].to(device, dtype = torch.float)
        object_mask = data['object_mask'].to(device, dtype = torch.float)

        # Entity Retriving Target
        entity_targets = data['target'].to(device, dtype = torch.float)
        # Convert start and end positions to torch.long
        start_positions = data['start_id'].to(device, dtype=torch.long)
        end_positions = data['end_id'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(input_ids, attention_mask, token_type_ids, pixel_values, bbox, encoded_token_objt_ids, bert_cls, visual_feat, norm_bbox, object_mask, positional_encoding)

        # Entity Retriving Task
        entity_logits = outputs_dict['entity_logits']
        entity_logits = entity_logits.squeeze(2)
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list = predict_entity_list + list(big_idx)

        _,target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list = target_id_list + list(target_idx)

        # Span-based QA Predicted Logits
        start_logits = outputs_dict['start_logits']
        end_logits = outputs_dict['end_logits']

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        # Define CrossEntropyLoss with ignore_index in the constructor
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2
        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
          pred_tokens = input_ids[i][pred_start_positions[i]:pred_end_positions[i] + 1]
          gt_tokens = input_ids[i][start_positions[i]:end_positions[i] + 1]

          # Decode tokens to text using processor
          pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
          gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

          # Calculate ANLS
          anls_score = calculate_anls(pred_text, gt_text)
          anls_scores.append(anls_score)

          predict_text_list.append(pred_text)
          target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    return average_anls, predict_text_list, target_text_list, predict_entity_list, target_id_list

## Updated Training Function with Grid Representations for Layout Pretraining
def train_layout(model,train_dataloader,lr):
    model.train()
    predict_text_list = []
    target_text_list = []
    anls_scores = []
    predict_entity_list = []
    target_id_list = []
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr) # change learning rate
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        # Convert tensors to the correct types
        input_ids = data['input_ids'].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data['attention_mask'].to(device, dtype=torch.float)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        bbox = data['bbox'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        encoded_token_objt_ids = data['token_objt_ids'].to(device, dtype=torch.long) ## For token aggregate into entities

        visual_feat = data['visual_feat'].to(device, dtype = torch.float)
        bert_cls = data['bert_cls'].to(device, dtype = torch.float)
        positional_encoding = data['positional_encoding'].to(device, dtype = torch.float)
        norm_bbox = data['norm_bbox'].to(device, dtype = torch.float)
        object_mask = data['object_mask'].to(device, dtype = torch.float)

        # Entity Retriving Target
        entity_targets = data['target'].to(device, dtype = torch.float)
        grid_emb = data['grid_emb'].to(device, dtype = torch.float)
        # Convert start and end positions to torch.long
        start_positions = data['start_id'].to(device, dtype=torch.long)
        end_positions = data['end_id'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(input_ids, attention_mask, token_type_ids, pixel_values, bbox, encoded_token_objt_ids, \
                             bert_cls, visual_feat, norm_bbox, object_mask, positional_encoding, grid_emb)

        # Entity Retriving Task
        entity_logits = outputs_dict['entity_logits']
        entity_logits = entity_logits.squeeze(2)
        loss_function = CrossEntropyLoss()
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list = predict_entity_list + list(big_idx)

        _,target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list = target_id_list + list(target_idx)

        # Span-based QA Predicted Logits
        start_logits = outputs_dict['start_logits']
        end_logits = outputs_dict['end_logits']

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        # Define CrossEntropyLoss with ignore_index in the constructor
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2
        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
          pred_tokens = input_ids[i][pred_start_positions[i]:pred_end_positions[i] + 1]
          gt_tokens = input_ids[i][start_positions[i]:end_positions[i] + 1]

          # Decode tokens to text using processor
          pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
          gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

          # Calculate ANLS
          anls_score = calculate_anls(pred_text, gt_text)
          anls_scores.append(anls_score)

          predict_text_list.append(pred_text)
          target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    return average_anls, predict_text_list, target_text_list, predict_entity_list, target_id_list

def train_layout_gemini(model,train_dataloader,lr):
    model.train()
    predict_text_list = []
    target_text_list = []
    anls_scores = []
    predict_entity_list = []
    target_id_list = []
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr) # change learning rate
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        # Convert tensors to the correct types
        input_ids = data['input_ids'].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data['attention_mask'].to(device, dtype=torch.float)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        bbox = data['bbox'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        encoded_token_objt_ids = data['token_objt_ids'].to(device, dtype=torch.long) ## For token aggregate into entities

        visual_feat = data['visual_feat'].to(device, dtype = torch.float)
        bert_cls = data['bert_cls'].to(device, dtype = torch.float)
        positional_encoding = data['positional_encoding'].to(device, dtype = torch.float)
        norm_bbox = data['norm_bbox'].to(device, dtype = torch.float)
        object_mask = data['object_mask'].to(device, dtype = torch.float)

        # Entity Retriving Target
        entity_targets = data['target'].to(device, dtype = torch.float)
        grid_emb = data['grid_emb'].to(device, dtype = torch.float)
        # Convert start and end positions to torch.long
        start_positions = data['start_id'].to(device, dtype=torch.long)
        end_positions = data['end_id'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(input_ids, attention_mask, token_type_ids, pixel_values, bbox, encoded_token_objt_ids, \
                             bert_cls, visual_feat, norm_bbox, object_mask, positional_encoding, grid_emb)

        # Entity Retriving Task
        entity_logits = outputs_dict['entity_logits']
        entity_logits = entity_logits.squeeze(2)
        loss_function = CrossEntropyLoss()
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list = predict_entity_list + list(big_idx)

        _,target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list = target_id_list + list(target_idx)

        # Span-based QA Predicted Logits
        start_logits = outputs_dict['start_logits']
        end_logits = outputs_dict['end_logits']

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        # Define CrossEntropyLoss with ignore_index in the constructor
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2
        total_loss = token_loss + entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
          pred_tokens = input_ids[i][pred_start_positions[i]:pred_end_positions[i] + 1]
          gt_tokens = input_ids[i][start_positions[i]:end_positions[i] + 1]

          # Decode tokens to text using processor
          pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
          gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

          # Calculate ANLS
          anls_score = calculate_anls(pred_text, gt_text)
          anls_scores.append(anls_score)

          predict_text_list.append(pred_text)
          target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    return average_anls, predict_text_list, target_text_list, predict_entity_list, target_id_list


def train_layout_coarse_grained(model,train_dataloader,lr):
    model.train()
    predict_text_list = []
    target_text_list = []
    anls_scores = []
    predict_entity_list = []
    target_id_list = []
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr) # change learning rate
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        # Convert tensors to the correct types
        input_ids = data['input_ids'].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data['attention_mask'].to(device, dtype=torch.float)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        bbox = data['bbox'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        encoded_token_objt_ids = data['token_objt_ids'].to(device, dtype=torch.long) ## For token aggregate into entities

        visual_feat = data['visual_feat'].to(device, dtype = torch.float)
        bert_cls = data['bert_cls'].to(device, dtype = torch.float)
        positional_encoding = data['positional_encoding'].to(device, dtype = torch.float)
        norm_bbox = data['norm_bbox'].to(device, dtype = torch.float)
        object_mask = data['object_mask'].to(device, dtype = torch.float)

        # Entity Retriving Target
        entity_targets = data['target'].to(device, dtype = torch.float)
        grid_emb = data['grid_emb'].to(device, dtype = torch.float)
        # Convert start and end positions to torch.long
        start_positions = data['start_id'].to(device, dtype=torch.long)
        end_positions = data['end_id'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs_dict = model(input_ids, attention_mask, token_type_ids, pixel_values, bbox, encoded_token_objt_ids, \
                             bert_cls, visual_feat, norm_bbox, object_mask, positional_encoding, grid_emb)

        # Entity Retriving Task
        entity_logits = outputs_dict['entity_logits']
        entity_logits = entity_logits.squeeze(2)
        loss_function = CrossEntropyLoss()
        entity_loss = loss_function(entity_logits, entity_targets)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list = predict_entity_list + list(big_idx)

        _,target_idx = torch.max(entity_targets.data, dim=1)
        target_id_list = target_id_list + list(target_idx)

        # Span-based QA Predicted Logits
        start_logits = outputs_dict['start_logits']
        end_logits = outputs_dict['end_logits']

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        # Compute loss with CrossEntropyLoss
        # Define CrossEntropyLoss with ignore_index in the constructor
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        token_loss = (start_loss + end_loss) / 2
        total_loss =  entity_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        pred_start_positions = torch.argmax(start_logits, dim=1)
        pred_end_positions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):  # Loop through each example in the batch
          pred_tokens = input_ids[i][pred_start_positions[i]:pred_end_positions[i] + 1]
          gt_tokens = input_ids[i][start_positions[i]:end_positions[i] + 1]

          # Decode tokens to text using processor
          pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
          gt_text = processor.tokenizer.decode(gt_tokens, skip_special_tokens=True)

          # Calculate ANLS
          anls_score = calculate_anls(pred_text, gt_text)
          anls_scores.append(anls_score)

          predict_text_list.append(pred_text)
          target_text_list.append(gt_text)

    # Calculate the average ANLS over all examples
    average_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    return average_anls, predict_text_list, target_text_list, predict_entity_list, target_id_list


def pretrain_layout(model, train_dataloader, lr):
    model.train()
    predict_entity_list = []
    target_id_list = []
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr) # change learning rate
    total_loss = 0
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        # Convert tensors to the correct types
        input_ids = data['input_ids'].to(device, dtype=torch.long).squeeze(1)
        attention_mask = data['attention_mask'].to(device, dtype=torch.float)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        bbox = data['bbox'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        encoded_token_objt_ids = data['token_objt_ids'].to(device, dtype=torch.long) ## For token aggregate into entities

        visual_feat = data['visual_feat'].to(device, dtype = torch.float)
        bert_cls = data['bert_cls'].to(device, dtype = torch.float)
        positional_encoding = data['positional_encoding'].to(device, dtype = torch.float)
        norm_bbox = data['norm_bbox'].to(device, dtype = torch.float)
        object_mask = data['object_mask'].to(device, dtype = torch.float)

        # Entity Retriving Target
        grid_emb = data['grid_emb'].to(device, dtype = torch.float)
        grid_idx = data['grid_ids'].to(device, dtype = torch.float)
        # Convert start and end positions to torch.long

        optimizer.zero_grad()

        # Forward pass through the model
        grid_logits = model(input_ids, attention_mask, token_type_ids, pixel_values, bbox, encoded_token_objt_ids,\
                             bert_cls, visual_feat, norm_bbox, object_mask, positional_encoding,grid_emb,train_stage = 'pretrain')

        # Entity Retriving Task
        entity_logits = grid_logits.squeeze(2)
        loss_fct =  CrossEntropyLoss()
        entity_loss = loss_fct(entity_logits, grid_idx)

        _, big_idx = torch.max(entity_logits.data, dim=1)
        predict_entity_list = predict_entity_list + list(big_idx)

        _,target_idx = torch.max(grid_idx.data, dim=1)
        target_id_list = target_id_list + list(target_idx)
        total_loss += entity_loss.item()  # Accumulate the loss

        # Backpropagation
        entity_loss.backward()
        optimizer.step()
    return predict_entity_list, target_id_list, total_loss

