"""
基于 ECharts 的离心泵故障知识图谱交互式可视化

用法:
    python visualize/echart_viz.py                    # 生成HTML
    python visualize/echart_viz.py --domain pump       # 默认
    python visualize/echart_viz.py --output graph.html # 指定输出
"""

import sys
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 配置 ──────────────────────────────────────────────
DEFAULT_OUTPUT = "pump_graph_viz.html"

# 实体类型 → 颜色（离心泵领域专门配色）
TYPE_COLORS = {
    "设备":       "#E74C3C",  # 红色 — 主要设备
    "部件":       "#E67E22",  # 橙色 — 设备部件
    "故障类型":   "#F1C40F",  # 黄色 — 故障
    "故障严重度": "#F39C12",  # 橙黄
    "传感器":     "#3498DB",  # 蓝色 — 测量设备
    "监测点":     "#2ECC71",  # 绿色 — 测点位置
    "测量记录":   "#1ABC9C",  # 青绿 — 时序数据
    "工况条件":   "#9B59B6",  # 紫色 — 运行条件
    "信号特征":   "#34495E",  # 深灰 — 信号分析
    "维修操作":   "#E91E63",  # 粉色 — 维修
}
FALLBACK_COLOR = "#95A5A6"


def load_data(data_dir: str = "data/output"):
    """加载实体和关系数据"""
    entities_path = os.path.join(data_dir, "entities.json")
    rels_path = os.path.join(data_dir, "relationships.json")

    entities = []
    if os.path.exists(entities_path):
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)

    relationships = []
    if os.path.exists(rels_path):
        with open(rels_path, "r", encoding="utf-8") as f:
            relationships = json.load(f)

    return entities, relationships


def build_graph_data(entities, relationships):
    """构建 ECharts graph 格式数据"""
    # 节点
    nodes = []
    seen_ids = set()
    for e in entities:
        eid = str(e.get("id", ""))
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)

        etype = str(e.get("type", "未知"))
        size = max(10, min(50, 12 + int(e.get("degree", 0)) * 1.5))

        nodes.append({
            "id": eid,
            "name": str(e.get("name", eid)),
            "symbolSize": size,
            "itemStyle": {"color": TYPE_COLORS.get(etype, FALLBACK_COLOR)},
            "category": etype,
            "entity_type": etype,
            "description": str(e.get("description", ""))[:300],
            "degree": int(e.get("degree", 0)),
            "source": str(e.get("source", "")),
        })

    # 分类
    categories = []
    seen_cats = set()
    for n in nodes:
        cat = n["category"]
        if cat not in seen_cats:
            seen_cats.add(cat)
            categories.append({
                "name": cat,
                "itemStyle": {"color": TYPE_COLORS.get(cat, FALLBACK_COLOR)},
            })

    # 关系
    links = []
    seen_links = set()
    for r in relationships:
        src = str(r.get("source", ""))
        tgt = str(r.get("target", ""))
        if not src or not tgt:
            continue
        link_key = f"{src}→{tgt}"
        if link_key in seen_links:
            continue
        seen_links.add(link_key)

        rel_type = str(r.get("relation", "关联"))
        weight = float(r.get("weight", 1.0))
        lw = max(0.5, min(5, weight * 2))

        links.append({
            "source": src,
            "target": tgt,
            "value": weight,
            "lineStyle": {"width": lw, "curveness": 0.15, "opacity": 0.4},
            "description": f"{rel_type}: {str(r.get('description', ''))[:200]}",
        })

    return {"nodes": nodes, "links": links, "categories": categories}


def generate_html(graph_data, output_path: str):
    """生成 ECharts HTML 可视化"""

    title = "离心泵故障诊断知识图谱"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"Microsoft YaHei","Segoe UI",sans-serif; background:#1a1a2e; overflow:hidden; }}
#header {{
    position:absolute; top:0; left:0; right:0; z-index:100;
    background:linear-gradient(135deg,#16213e,#0f3460);
    padding:10px 24px; display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 2px 16px rgba(0,0,0,0.5); color:#eee;
}}
#header h1 {{ font-size:18px; font-weight:600; }}
#header .stats {{ font-size:12px; color:#aab; }}
#header .stats span {{ margin:0 10px; }}
#chart {{ position:absolute; top:48px; left:0; right:0; bottom:0; }}
#detail-panel {{
    position:absolute; top:70px; right:16px; z-index:100; width:340px;
    max-height:calc(100vh - 100px); overflow-y:auto;
    background:rgba(22,33,62,0.96); border:1px solid #2a3a5e;
    border-radius:8px; padding:18px; color:#ddd;
    display:none; box-shadow:0 6px 24px rgba(0,0,0,0.5);
}}
#detail-panel h3 {{ color:#fff; margin-bottom:6px; font-size:16px; }}
#detail-panel .type-badge {{
    display:inline-block; padding:2px 10px; border-radius:10px;
    font-size:11px; font-weight:600; color:#fff; margin-bottom:10px;
}}
#detail-panel p {{ font-size:12px; line-height:1.6; margin-bottom:4px; color:#bbb; }}
#detail-panel .close-btn {{
    position:absolute; top:8px; right:14px; cursor:pointer;
    font-size:18px; color:#888; background:none; border:none;
}}
#detail-panel .close-btn:hover {{ color:#fff; }}
.desc-header {{ font-size:11px; color:#888; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; }}
.desc-body {{ font-size:12px; line-height:1.7; color:#bbb; max-height:200px; overflow-y:auto;
    background:rgba(255,255,255,0.03); border-radius:6px; padding:10px; white-space:pre-wrap; word-break:break-word; }}
#legend-info {{
    position:absolute; bottom:8px; left:12px; z-index:100;
    color:#666; font-size:11px;
}}
</style>
</head>
<body>

<div id="header">
    <h1>{title}</h1>
    <div class="stats">
        <span>实体: {len(graph_data["nodes"])}</span>
        <span>关系: {len(graph_data["links"])}</span>
        <span>类型: {len(graph_data["categories"])}</span>
    </div>
</div>

<div id="chart"></div>

<div id="detail-panel">
    <button class="close-btn" onclick="closeDetail()">x</button>
    <h3 id="detail-title"></h3>
    <div id="detail-type-badge"></div>
    <div class="desc-section">
        <div class="desc-header">描述</div>
        <div class="desc-body" id="detail-desc"></div>
    </div>
</div>

<div id="legend-info">滚轮缩放 | 拖拽节点 | 点击查看详情 | 图例筛选</div>

<script>
(function() {{
    var data = {json.dumps(graph_data["nodes"], ensure_ascii=False)};
    var links = {json.dumps(graph_data["links"], ensure_ascii=False)};
    var categories = {json.dumps(graph_data["categories"], ensure_ascii=False)};

    var chartDom = document.getElementById('chart');
    var myChart = echarts.init(chartDom);

    var option = {{
        tooltip: {{
            trigger: 'item',
            formatter: function(p) {{
                if (p.dataType === 'edge' && p.data) {{
                    return '<b>' + p.data.source + ' → ' + p.data.target + '</b><br/>' +
                           p.data.description || '';
                }}
                if (p.dataType === 'node' && p.data) {{
                    return '<b>' + p.data.name + '</b><br/>' +
                           '类型: ' + p.data.entity_type + '<br/>' +
                           '连接数: ' + p.data.degree + '<br/>' +
                           '<small style="color:#aaa">' + (p.data.description||'') + '</small>';
                }}
                return '';
            }},
            backgroundColor: 'rgba(20,30,50,0.95)',
            borderColor: '#3a4a6e',
            textStyle: {{ color: '#ddd', fontSize: 12 }}
        }},
        legend: {{
            data: categories.map(function(c) {{ return c.name; }}),
            bottom: 6,
            textStyle: {{ color: '#aaa', fontSize: 11 }},
            selectedMode: 'multiple'
        }},
        series: [{{
            type: 'graph',
            layout: 'force',
            data: data,
            links: links,
            categories: categories,
            roam: true,
            draggable: true,
            force: {{
                repulsion: 800,
                gravity: 0.1,
                edgeLength: [100, 300],
                layoutAnimation: true,
                friction: 0.6
            }},
            emphasis: {{
                focus: 'adjacency',
                lineStyle: {{ width: 5 }}
            }},
            scaleLimit: {{ min: 0.15, max: 6 }},
            lineStyle: {{ color: 'rgba(150,150,150,0.3)', curveness: 0.15 }},
            label: {{
                show: true,
                position: 'right',
                fontSize: 10,
                color: '#ccc',
                formatter: function(p) {{
                    return p.name.length > 18 ? p.name.substring(0, 16) + '...' : p.name;
                }}
            }},
            edgeSymbol: ['none', 'none']
        }}]
    }};

    myChart.setOption(option);

    // 点击 → 详情面板
    myChart.on('click', function(params) {{
        if (params.dataType === 'node' && params.data) {{
            var panel = document.getElementById('detail-panel');
            document.getElementById('detail-title').textContent = params.data.name;
            var badge = document.getElementById('detail-type-badge');
            badge.textContent = params.data.entity_type || '';
            badge.style.background = (params.data.itemStyle && params.data.itemStyle.color) || '#888';
            document.getElementById('detail-desc').textContent = params.data.description || '暂无描述';
            panel.style.display = 'block';
        }} else {{
            closeDetail();
        }}
    }});

    window.closeDetail = function() {{
        document.getElementById('detail-panel').style.display = 'none';
        myChart.dispatchAction({{ type: 'downplay', seriesIndex: 0 }});
    }};

    window.addEventListener('resize', function() {{ myChart.resize(); }});
}})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[ECharts] 知识图谱已生成: {output_path} ({size_kb:.0f} KB)")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="离心泵知识图谱 ECharts 可视化")
    parser.add_argument("--data-dir", default="data/output",
                        help="数据目录 (默认: data/output)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"输出HTML (默认: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print("[1/3] 加载数据...")
    entities, relationships = load_data(args.data_dir)
    print(f"   实体: {len(entities)}, 关系: {len(relationships)}")

    print("[2/3] 构建图数据...")
    graph_data = build_graph_data(entities, relationships)
    print(f"   节点: {len(graph_data['nodes'])}, 链接: {len(graph_data['links'])}")

    print("[3/3] 生成 HTML...")
    generate_html(graph_data, args.output)


if __name__ == "__main__":
    main()
