# main.py
#!/usr/bin/env python3
"""
SLAM系统参数自动调优与标定Agent主程序
"""

import sys
import signal
import traceback
from main_agent_system import SLAMAutoTuningAgent

def signal_handler(sig, frame):
    """信号处理器"""
    print('\nShutting down SLAM Auto Tuning Agent...')
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建并启动Agent
        agent = SLAMAutoTuningAgent()
        
        print("Starting SLAM Auto Tuning Agent...")
        print("Environment detection and parameter optimization will begin automatically.")
        print("Press Ctrl+C to stop.")
        
        # 启动主运行循环
        agent.run()
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"Fatal error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
