def llm_deep_analysis(payload: dict) -> dict:
    """
    使用 GPT-5 Responses API 进行深度分析。
    新版 API 不支持 response_format，只能在 prompt 里强制模型输出 JSON。
    """
    if client is None:
        return {
            "overall_summary": "未配置 OPENAI_API_KEY 或未安装 openai SDK，当前仅展示规则引擎结果，未启用 AI 深度分析。",
            "key_findings": [],
            "prioritized_actions": [],
            "risks": [],
            "data_gaps": []
        }

    prompt = f"""
你是一名北美餐饮 & 外卖运营专家，熟悉 DoorDash、UberEats、中餐厅经营。

下面是该餐厅的结构化 JSON 信息：
{json.dumps(payload, ensure_ascii=False, indent=2)}

请基于以上信息输出餐厅的深度诊断结果。

⚠️ 输出格式必须是 严格 JSON，不要出现多余文字、不允许加解释、不允许 Markdown。

固定输出 JSON schema 如下：

{{
  "overall_summary": "string，整体总结",
  "key_findings": ["string 列表，核心洞察"],
  "prioritized_actions": [
    {{
      "horizon": "short_term 或 mid_term",
      "description": "行动建议"
    }}
  ],
  "risks": ["string 列表，主要风险点"],
  "data_gaps": ["string 列表，缺失的数据"]
}}

只返回 JSON，不能出现代码块、注释、额外说明。
"""

    resp = client.responses.create(
        model="gpt-5.1-mini",
        input=prompt,
        max_output_tokens=1500
    )

    raw_output = resp.output_text

    # 尝试解析 JSON
    try:
        return json.loads(raw_output)
    except Exception:
        try:
            fixed = raw_output[raw_output.find("{"): raw_output.rfind("}") + 1]
            return json.loads(fixed)
        except Exception:
            return {
                "overall_summary": raw_output,
                "key_findings": [],
                "prioritized_actions": [],
                "risks": [],
                "data_gaps": []
            }