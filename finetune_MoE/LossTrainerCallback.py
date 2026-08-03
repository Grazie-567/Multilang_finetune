import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import json
from datetime import datetime
import os

# ==================== 新增Loss追踪回调类 ====================

class LossTrackerCallback(TrainerCallback):
    """追踪并记录训练与验证loss，支持训练结束后绘图"""
    
    def __init__(self, output_dir: str, plot_after_train: bool = True):
        self.output_dir = output_dir
        self.loss_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "steps": [],
            "epochs": []
        }
        self.plot_after_train = plot_after_train
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否是主进程
        self.is_main_process = (int(os.environ.get("RANK", 0)) == 0)
        if self.is_main_process:
            print(f"📊 LossTracker: 主进程(Rank 0)将负责记录和绘图")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process or logs is None:
            return
        # 检查是否是累积步的最后一步
        is_update_step = (state.global_step % args.gradient_accumulation_steps == 0)
        if not is_update_step and "loss" in logs:
            return  # 跳过中间累积步骤，只记录参数更新时的loss
        
        # 记录到内存
        if state.is_world_process_zero:  # 仅在主进程记录
            current_step = state.global_step
            
            # 捕获训练loss（可能每几步记录一次）
            if "loss" in logs:
                self.loss_history["train_loss"].append(logs["loss"])
                self.loss_history["steps"].append(current_step)
                self.loss_history["epochs"].append(state.epoch)
                print(f"📈 Step {current_step} - Train Loss: {logs['loss']:.4f}")  # 只在rank 0打印
            
            # 捕获验证loss（在eval时记录）
            if "eval_loss" in logs:
                self.loss_history["eval_loss"].append(logs["eval_loss"])
            
            # 捕获学习率
            if "learning_rate" in logs:
                self.loss_history["learning_rate"].append(logs["learning_rate"])
            
            # 实时保存到CSV（防止意外中断丢失数据）
            self._save_to_csv()
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时绘制曲线"""
        if self.is_main_process and self.plot_after_train:
            print("\n📊 正在生成loss曲线图...")
            self._plot_loss_curves()
            print(f"✅ Loss曲线已保存至: {self.output_dir}")
    
    def _save_to_csv(self):
        """将loss历史保存到CSV文件"""
        if not self.is_main_process:
            return
        csv_path = os.path.join(self.output_dir, "loss_history.csv")
        
        # 创建DataFrame（处理不同长度的序列）
        max_len = max(len(self.loss_history["train_loss"]), 
                      len(self.loss_history["eval_loss"]))
        
        # 对齐数据长度（用None填充）
        data = {
            "step": self.loss_history["steps"],
            "epoch": self.loss_history["epochs"],
            "train_loss": self.loss_history["train_loss"] + [None] * (max_len - len(self.loss_history["train_loss"])),
            "eval_loss": self.loss_history["eval_loss"] + [None] * (max_len - len(self.loss_history["eval_loss"])),
            "learning_rate": self.loss_history["learning_rate"] + [None] * (max_len - len(self.loss_history["learning_rate"]))
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        # 同时保存JSON格式（便于后续分析）
        json_path = os.path.join(self.output_dir, "loss_history.json")
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def _plot_loss_curves(self, smooth_window: int = 5):
        """绘制loss曲线图"""
        if not self.is_main_process or not self.loss_history["steps"]:
            return
        
        # 设置中文字体支持（如果需要）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表（2x2子图）
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Whisper MoE Fine-tuning Loss Curves', fontsize=16)
        
        # 1. 训练Loss曲线
        ax1 = axes[0, 0]
        if self.loss_history["train_loss"]:
            steps = self.loss_history["steps"][:len(self.loss_history["train_loss"])]
            # 平滑处理
            if len(self.loss_history["train_loss"]) > smooth_window:
                smoothed = pd.Series(self.loss_history["train_loss"]).rolling(smooth_window).mean()
                ax1.plot(steps, smoothed, 'b-', linewidth=2, label=f'Train Loss (smooth={smooth_window})')
            ax1.plot(steps, self.loss_history["train_loss"], 'b-', alpha=0.3, label='Raw Train Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 验证Loss曲线
        ax2 = axes[0, 1]
        if self.loss_history["eval_loss"]:
            # 验证loss的步数（通常是eval_steps的倍数）
            eval_steps = self.loss_history["steps"][:len(self.loss_history["eval_loss"])]
            ax2.plot(eval_steps, self.loss_history["eval_loss"], 'r-o', 
                    linewidth=2, markersize=5, label='Eval Loss')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        ax3 = axes[1, 0]
        if self.loss_history["learning_rate"]:
            lr_steps = self.loss_history["steps"][:len(self.loss_history["learning_rate"])]
            ax3.plot(lr_steps, self.loss_history["learning_rate"], 'g-', linewidth=2)
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.grid(True, alpha=0.3)
        
        # 4. Train & Eval Loss对比
        ax4 = axes[1, 1]
        if self.loss_history["train_loss"]:
            ax4.plot(self.loss_history["steps"][:len(self.loss_history["train_loss"])], 
                    self.loss_history["train_loss"], 'b-', alpha=0.5, label='Train Loss')
        if self.loss_history["eval_loss"]:
            ax4.plot(self.loss_history["steps"][:len(self.loss_history["eval_loss"])], 
                    self.loss_history["eval_loss"], 'r-o', label='Eval Loss')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss')
        ax4.set_title('Train vs Eval Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, "loss_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # 保存PDF矢量图（用于论文）
        pdf_path = os.path.join(self.output_dir, "loss_curves.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close()  # 释放内存