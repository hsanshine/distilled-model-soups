import sys
import os
import time
import torch
import json
import numpy as np
import random

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None, term_width: int = None):
    if term_width is None:
        _, term_width = os.popen('stty size', 'r').read().split()
        term_width = int(term_width)
    else:
        term_width = term_width

    TOTAL_BAR_LENGTH = 65.

    global last_time, begin_time
    
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n\n\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def train(
    model, dataloader, criterion, optimizer, teacher_model=None, 
    distill_criterion=None, alpha=0.1, temperature=10, loss_method='method1', device='cpu'
):
    if teacher_model!=None:
        teacher_model.eval()
        
    correct = 0 
    total = 0
    total_loss = 0
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # predict
        outputs = model(inputs)
        
        if teacher_model != None:
            teacher_outputs = teacher_model(inputs)
            
        # loss and update
        optimizer.zero_grad()
                
        loss = criterion(outputs, targets)
        
        # Knowledge Distillation
        if teacher_model != None:
            if loss_method == 'method1':
                student_outputs = torch.nn.functional.log_softmax(outputs / temperature, dim=1) # log softmax
                teacher_outputs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1) # softmax
                
                distill_loss = distill_criterion(student_outputs, teacher_outputs)
                
                loss = (1 - alpha) * loss + (alpha * distill_loss)

            elif loss_method == 'method2':
                student_outputs = torch.nn.functional.log_softmax(outputs, dim=1) # log softmax
                teacher_outputs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1) # softmax
                
                distill_loss = distill_criterion(student_outputs, teacher_outputs)
                
                loss = (1 - alpha) * loss + (alpha * distill_loss)
            
            elif loss_method == 'method3':
                student_outputs = torch.nn.functional.log_softmax(outputs / temperature, dim=1) # log softmax
                teacher_outputs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1) # softmax
                
                distill_loss = distill_criterion(student_outputs, teacher_outputs)
                
                loss = (1 - alpha) * loss + (2 * alpha * temperature**2 * distill_loss)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ')
            exit(1)
            
        loss.backward()
        optimizer.step()
        
        # total loss and acc
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=1) 
        correct += targets.eq(preds).sum().item()
        total += targets.size(0)
        
        # massage
        progress_bar(current=idx, 
                     total=len(dataloader),
                     msg='Loss: %.3f | Acc: %.3f%% [%d/%d]' % (total_loss/(idx + 1), 
                                                               100.*(correct/total), correct, total),
                     term_width=100)

    return correct/total, total_loss/len(dataloader)
        

def test(model, dataloader, criterion, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    best_acc = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            # massage
            progress_bar(current=idx, 
                         total=len(dataloader),
                         msg='Loss: %.3f | Acc: %.3f%% [%d/%d]' % (total_loss/(idx + 1), 
                                                                 100.*(correct/total), correct, total),
                         term_width=100)

    return correct/total, total_loss/len(dataloader)

            
def fit(
    model, start_epoch, epochs, trainloader, testloader, criterion, optimizer, scheduler, savedir, writer,
    teacher=None, distill_criterion=None, alpha=0.1, temperature=10, temperature_scheduler=False,
    loss_method='method1', device='cpu'
):

    best_acc = 0
    if temperature_scheduler:
        temp_step = (temperature-1)/epochs
    else:
        temp_step = 0

    for epoch in range(start_epoch, epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        temperature = temperature - (epoch*temp_step)
        train_acc, train_loss = train(model, trainloader, criterion, optimizer, 
                                      teacher, distill_criterion, alpha, temperature, loss_method, device)
        valid_acc, valid_loss = test(model, testloader, criterion, device)

        scheduler.step()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('Acc/Validation', valid_acc, epoch)
    
        # checkpoint
        if best_acc < valid_acc:
            state = {'best_epoch':epoch, 'best_acc':valid_acc}
            json.dump(state, open(savedir.replace('pt','json'),'w'), indent=4)

            weights = {'model':model.state_dict()}
            torch.save(weights, savedir)
            
            print('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, valid_acc))

            best_acc = valid_acc