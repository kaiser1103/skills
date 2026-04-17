#!/usr/bin/env python3
"""
CPU RAG 插件测试脚本

用于验证插件是否正常工作。
"""

import sys
import os

# 添加 hermes-agent 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_import():
    """测试模块导入"""
    print("📝 测试模块导入...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        print("  ✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def test_provider_creation():
    """测试 Provider 创建"""
    print("📝 测试 Provider 创建...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        provider = CPURAGMemoryProvider()
        assert provider.name == "cpu_rag"
        print(f"  ✓ Provider 创建成功: {provider.name}")
        return True
    except Exception as e:
        print(f"  ✗ 创建失败: {e}")
        return False

def test_config_schema():
    """测试配置模式"""
    print("📝 测试配置模式...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        provider = CPURAGMemoryProvider()
        schema = provider.get_config_schema()
        assert len(schema) > 0
        print(f"  ✓ 配置模式: {len(schema)} 个字段")
        for field in schema:
            print(f"    - {field['key']}: {field.get('default', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ✗ 配置模式失败: {e}")
        return False

def test_tool_schemas():
    """测试工具模式"""
    print("📝 测试工具模式...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        provider = CPURAGMemoryProvider()
        tools = provider.get_tool_schemas()
        assert len(tools) == 3
        tool_names = [t['name'] for t in tools]
        assert 'rag_search' in tool_names
        assert 'rag_add_memory' in tool_names
        assert 'rag_stats' in tool_names
        print(f"  ✓ 工具模式: {tool_names}")
        return True
    except Exception as e:
        print(f"  ✗ 工具模式失败: {e}")
        return False

def test_availability():
    """测试依赖可用性"""
    print("📝 测试依赖可用性...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        provider = CPURAGMemoryProvider()
        available = provider.is_available()
        
        if available:
            print("  ✓ 所有依赖已安装")
        else:
            print("  ⚠️  依赖缺失，请运行 install.sh 安装")
        return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

def test_system_prompt():
    """测试系统提示词"""
    print("📝 测试系统提示词...")
    try:
        from plugins.memory.cpu_rag import CPURAGMemoryProvider
        provider = CPURAGMemoryProvider()
        prompt = provider.system_prompt_block()
        assert "CPU RAG" in prompt
        print(f"  ✓ 系统提示词正确")
        print(f"    {prompt[:100]}...")
        return True
    except Exception as e:
        print(f"  ✗ 系统提示词失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 CPU RAG 插件测试")
    print("=" * 50)
    print()
    
    tests = [
        test_import,
        test_provider_creation,
        test_config_schema,
        test_tool_schemas,
        test_availability,
        test_system_prompt,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    print(f"结果: {sum(results)}/{len(results)} 个测试通过")
    
    if all(results):
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
