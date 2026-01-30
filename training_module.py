def train_with_val_safe(models_dict, train_loader, test_loader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    all_history = {}

    for name, model in models_dict.items():
        print(f"\n 選手入場: {name}")
        model.to(device)
        
        # 優化器建議用 AdamW 搭配重量衰減
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()
        
        start_epoch = 0
        best_test_acc = 0.0
        history = {'train_acc': [], 'test_acc': [], 'loss': [], 'l1': []}
        
        checkpoint_path = os.path.join(save_dir, f"{name}_resume.pth")
        
        # 嘗試讀取續傳存檔
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                best_test_acc = ckpt.get('best_test_acc', 0.0)
                history = ckpt.get('history', history)
                print(f"偵測到存檔，從 Epoch {start_epoch} 恢復。目前最佳測試 Acc: {best_test_acc:.4f}")
            except Exception as e:
                print(f" 讀取存檔失敗，將重新開始。原因: {e}")

        # 定義統一保存邏輯
        def save_now(current_epoch, is_best=False):
            state = {
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_acc': best_test_acc,
                'history': history
            }
            torch.save(state, checkpoint_path)
            if is_best:
                torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_best_real.pth"))

        try:
            for epoch in range(start_epoch, epochs):
                # --- 訓練階段 ---
                model.train()
                train_correct, train_total, ep_loss, ep_l1 = 0, 0, 0, 0
                pbar = tqdm.tqdm(train_loader, desc=f"{name} Ep {epoch+1}")
                
                for imgs, lbls in pbar:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    optimizer.zero_grad()
                    with autocast():
                        out = model(imgs)
                        ce_loss = F.cross_entropy(out, lbls)
                        # L1 正則化 (如果你需要保持模型稀疏)
                        l1_val = sum(p.abs().sum() for p in model.parameters())
                        total_loss = ce_loss + 1e-5 * l1_val
                    
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_correct += (out.argmax(1) == lbls).sum().item()
                    train_total += lbls.size(0)
                    ep_loss += ce_loss.item()
                    ep_l1 += l1_val.item()
                    pbar.set_postfix({"TrainAcc": f"{train_correct/train_total:.4f}"})

                # --- 驗證階段 (for early stop ) ---
                model.eval()
                test_correct, test_total = 0, 0
                with torch.no_grad():
                    for t_imgs, t_lbls in test_loader:
                        t_imgs, t_lbls = t_imgs.to(device), t_lbls.to(device)
                        t_out = model(t_imgs)
                        test_correct += (t_out.argmax(1) == t_lbls).sum().item()
                        test_total += t_lbls.size(0)
                
                cur_test_acc = test_correct / test_total
                scheduler.step()
                
                # 紀錄歷史
                history['train_acc'].append(train_correct / train_total)
                history['test_acc'].append(cur_test_acc)
                history['loss'].append(ep_loss / len(train_loader))
                history['l1'].append(ep_l1 / len(train_loader))
                # for early stop 
                is_best = cur_test_acc > best_test_acc
                if is_best:
                    best_test_acc = cur_test_acc
                
                # forced saved 
                save_now(epoch, is_best=is_best)
                print(f"📉 Ep {epoch+1} 完成! Train Acc: {history['train_acc'][-1]:.4f} | Test Acc: {cur_test_acc:.4f}")

        except KeyboardInterrupt:
            print("\n🛑 偵測到手動中斷！正在緊急保存目前進度...")
            save_now(epoch) 
            print("💾 緊急存檔成功！下次執行將自動接續。")
            break
        
        all_history[name] = history
    return all_history
