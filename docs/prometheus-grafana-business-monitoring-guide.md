# Prometheus 与 Grafana 接入实践：从系统指标到业务监控的通用指南

## 引言

很多项目在早期只依赖日志排查问题：接口报错了看日志，任务失败了看日志，用户反馈慢了再去翻调用链。日志当然重要，但当系统进入持续运行阶段，仅靠日志很难回答一些趋势性问题：

- 最近 5 分钟请求量是否突然升高？
- 接口 P95 延迟是否持续变差？
- 后台任务是否堆积？
- 外部依赖调用失败率是否异常？
- 某个业务流程的成功率是否下降？
- 用户反馈、订单转化、消息处理、支付回调等关键业务指标是否稳定？

Prometheus 和 Grafana 的组合，正是为这类问题提供答案的常见方案。Prometheus 负责采集、存储和查询时间序列指标；Grafana 负责把指标组织成仪表盘、趋势图和告警视图。无论项目是电商系统、IM 服务、支付系统、AI 应用、后台管理平台，还是普通 Web API，只要它有长期运行和稳定性要求，就可以通过这套方法建立可观测能力。

本文不绑定任何具体项目代码，而是从通用工程实践出发，介绍如何为一个服务接入 Prometheus 和 Grafana，并进一步从基础系统监控扩展到业务监控。

## 一、先理解 Prometheus 和 Grafana 分别解决什么问题

Prometheus 的核心模型是“拉取指标”。应用在某个 HTTP 端点暴露指标，例如 `/metrics`，Prometheus 按固定周期访问这个端点，把返回内容存成时间序列。

Grafana 不负责采集指标，它更像一个可视化和分析入口。Grafana 连接 Prometheus 数据源后，可以使用 PromQL 查询指标，并把结果展示成折线图、柱状图、仪表盘、表格或告警规则。

可以简单理解为：

```text
应用服务
  └─ 暴露 /metrics
        ↓ Prometheus 定时抓取
Prometheus
  └─ 存储时间序列，支持 PromQL 查询
        ↓ Grafana 查询
Grafana
  └─ 展示仪表盘、趋势图和告警视图
```

这套架构的优点是清晰、通用、侵入性低。应用只需要暴露标准格式的指标，Prometheus 和 Grafana 就可以独立部署和演进。

## 二、监控不只是 CPU 和内存

接入监控时，最容易先想到 CPU、内存、磁盘和网络。这些指标属于基础设施监控，它们回答的是“机器或容器是否健康”。

但业务系统真正需要回答的问题通常更具体：

- 接口是否慢了？
- 任务是否堆积了？
- 数据同步是否失败了？
- 缓存命中率是否下降了？
- 支付回调、消息投递、订单创建是否成功？
- 第三方接口是否变慢或不稳定？

因此，一个成熟的监控体系通常分为三层：

```text
基础设施监控：CPU、内存、磁盘、网络、容器状态
应用运行监控：HTTP 请求量、延迟、错误率、线程/连接/队列状态
业务监控：订单量、支付成功率、消息处理量、任务成功率、缓存命中率、用户反馈
```

Prometheus 和 Grafana 可以同时覆盖这三层。系统指标通常由 node_exporter、cAdvisor、Kubernetes 组件或云厂商监控提供；应用和业务指标则需要在代码中主动埋点。

## 三、Prometheus 的四类指标

Prometheus 常见指标类型包括 Counter、Gauge、Histogram 和 Summary。实际项目中，最常用的是 Counter、Gauge 和 Histogram。

### 1. Counter：只增不减的累计值

Counter 适合记录“发生了多少次”。例如：

```text
http_requests_total
order_created_total
payment_callback_total
message_consumed_total
job_completed_total
external_api_call_total
```

Counter 不能减少，通常配合 `rate()` 或 `increase()` 使用。

示例：

```promql
sum(rate(http_requests_total[5m]))
```

表示最近 5 分钟每秒请求速率。

```promql
sum(increase(order_created_total[1h]))
```

表示过去 1 小时新增订单数。

### 2. Gauge：可增可减的当前值

Gauge 适合记录“当前状态”。例如：

```text
active_requests
queue_size
running_jobs
online_users
open_connections
```

它可以增加，也可以减少。比如当前正在执行的任务数，任务开始时加一，任务结束时减一。

### 3. Histogram：观察值分布

Histogram 适合记录耗时、大小等分布数据。例如：

```text
http_request_duration_seconds
db_query_duration_seconds
external_api_duration_seconds
job_duration_seconds
```

Histogram 会把观测值放进不同 bucket，随后可以用 `histogram_quantile()` 计算 P50、P95、P99。

示例：

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)
```

这表示最近 5 分钟 HTTP 请求延迟的 P95。

### 4. Summary：客户端侧分位数

Summary 也可以用于观察分位数，但在服务端聚合方面不如 Histogram 灵活。多数业务服务中，优先使用 Histogram 更容易和 Grafana、PromQL 聚合查询配合。

## 四、在应用中暴露 `/metrics`

不同语言都有 Prometheus 客户端库。下面以 Python 的 FastAPI 为例，展示一个最小接入方式。其他语言如 Java、Go、Node.js 的思路类似：定义指标、在业务代码中记录指标、暴露 `/metrics`。

```python
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

app = FastAPI()

request_total = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
```

还可以通过中间件记录 HTTP 请求耗时：

```python
import time
from prometheus_client import Histogram

http_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

@app.middleware("http")
async def record_http_metrics(request, call_next):
    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        http_duration.labels(
            method=request.method,
            path=request.url.path,
            status=status,
        ).observe(time.perf_counter() - started)
```

这只是入口层监控。真正有价值的业务监控，还需要在核心业务流程中埋点。

## 五、业务指标应该如何设计

业务指标设计的核心原则是：先明确问题，再定义指标。不要为了“指标多”而埋点，而是围绕系统最关键的业务路径进行建模。

假设一个系统中存在以下业务流程：

```text
请求进入
  -> 参数校验
  -> 查询缓存
  -> 访问数据库
  -> 调用外部服务
  -> 执行业务决策
  -> 写入结果
  -> 返回响应
```

可以设计以下几组指标。

### 1. 请求吞吐与错误率

```text
http_requests_total{method, path, status}
```

常用查询：

```promql
sum(rate(http_requests_total[5m]))
```

```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
clamp_min(sum(rate(http_requests_total[5m])), 0.001)
```

前者表示请求速率，后者表示 5xx 错误率。

### 2. 接口延迟

```text
http_request_duration_seconds_bucket{method, path, status, le}
```

常用查询：

```promql
histogram_quantile(
  0.95,
  sum by (le, path) (rate(http_request_duration_seconds_bucket[5m]))
)
```

这可以看出不同接口的 P95 延迟。如果一个接口平均耗时不高，但 P99 很高，说明部分请求存在长尾问题。

### 3. 核心业务流程耗时

对于订单创建、支付处理、消息投递、文件转换、模型推理、数据同步等流程，建议单独记录业务流程耗时：

```text
business_flow_duration_seconds{flow_name, status}
```

示例：

```python
from prometheus_client import Histogram
import time

flow_duration = Histogram(
    "business_flow_duration_seconds",
    "Business flow duration in seconds.",
    ["flow_name", "status"],
)

def run_business_flow(flow_name: str, func):
    started = time.perf_counter()
    status = "success"
    try:
        return func()
    except Exception:
        status = "error"
        raise
    finally:
        flow_duration.labels(
            flow_name=flow_name,
            status=status,
        ).observe(time.perf_counter() - started)
```

这类指标比 HTTP 耗时更贴近业务。因为同一个 HTTP 接口内部可能包含多个阶段，只有拆分业务流程，才能知道慢点究竟在哪里。

### 4. 阶段级指标

复杂流程可以继续拆成阶段：

```text
business_stage_duration_seconds{flow_name, stage_name}
business_stage_total{flow_name, stage_name, status}
```

例如：

```text
flow_name="order_create", stage_name="check_inventory"
flow_name="order_create", stage_name="create_payment"
flow_name="order_create", stage_name="write_order"
```

或：

```text
flow_name="message_delivery", stage_name="route_message"
flow_name="message_delivery", stage_name="push_gateway"
flow_name="message_delivery", stage_name="ack_wait"
```

Grafana 中可以用一个面板展示每个阶段的 P95 延迟，快速定位瓶颈。

### 5. 外部依赖指标

只看本服务自身并不够。很多系统故障来自外部依赖，例如数据库、缓存、消息队列、对象存储、支付网关、短信服务、模型服务等。

建议指标：

```text
external_call_total{provider, operation, status}
external_call_duration_seconds{provider, operation}
```

查询外部调用失败率：

```promql
sum by (provider) (rate(external_call_total{status!="success"}[5m]))
/
clamp_min(sum by (provider) (rate(external_call_total[5m])), 0.001)
```

这可以帮助区分“业务代码变慢”和“外部依赖变慢”。

### 6. 缓存、队列和异步任务

常见指标包括：

```text
cache_hit_total{cache_name}
cache_miss_total{cache_name}
queue_size{queue_name}
job_total{job_name, status}
job_duration_seconds{job_name}
```

缓存命中率：

```promql
sum(rate(cache_hit_total[5m]))
/
clamp_min(
  sum(rate(cache_hit_total[5m])) + sum(rate(cache_miss_total[5m])),
  0.001
)
```

异步任务失败率：

```promql
sum(rate(job_total{status="failed"}[5m]))
/
clamp_min(sum(rate(job_total[5m])), 0.001)
```

队列长度和任务耗时经常需要一起看：队列长度升高但任务耗时不变，可能是流量突增；队列长度升高且任务耗时升高，可能是下游处理变慢。

## 六、Prometheus 抓取配置

应用暴露 `/metrics` 后，需要配置 Prometheus 抓取。

一个最小的 `prometheus.yml`：

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "my-service"
    metrics_path: "/metrics"
    static_configs:
      - targets:
          - "app:8000"
```

如果本地使用 Docker Compose，可以这样组织：

```yaml
services:
  app:
    image: your-app:latest
    ports:
      - "8000:8000"

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    depends_on:
      - app

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

启动后可以访问：

```text
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
应用指标:   http://localhost:8000/metrics
```

生产环境中不建议把 `/metrics` 直接暴露到公网。更常见的做法是让 Prometheus 在内网访问，或通过 Kubernetes ServiceMonitor、PodMonitor、服务发现等方式自动发现目标。

## 七、Grafana 仪表盘应该如何组织

Grafana 面板不应该简单堆指标，而应该按照排障路径组织。

### 1. 总览层

总览层回答“系统现在是否正常”：

- 请求速率
- 错误率
- P95 / P99 延迟
- 当前活跃任务数
- 队列长度
- 核心业务成功率

示例面板：

```promql
sum(rate(http_requests_total[5m]))
```

```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
clamp_min(sum(rate(http_requests_total[5m])), 0.001)
```

### 2. 接口层

接口层回答“哪个入口异常”：

- 按 path 统计请求量
- 按 path 统计错误率
- 按 path 统计 P95 延迟

```promql
topk(10, sum by (path) (rate(http_requests_total[5m])))
```

```promql
histogram_quantile(
  0.95,
  sum by (le, path) (rate(http_request_duration_seconds_bucket[5m]))
)
```

### 3. 业务流程层

业务流程层回答“慢在哪个业务阶段”：

- 各业务流程成功 / 失败次数
- 各业务流程耗时分位数
- 各业务阶段耗时分位数
- 关键业务量趋势

```promql
sum by (flow_name, status) (
  increase(business_flow_total[1h])
)
```

```promql
histogram_quantile(
  0.95,
  sum by (le, flow_name, stage_name) (
    rate(business_stage_duration_seconds_bucket[5m])
  )
)
```

### 4. 依赖层

依赖层回答“是不是下游拖慢了系统”：

- 数据库查询耗时
- 缓存命中率
- 消息队列堆积
- 第三方接口错误率
- 外部服务 P95 延迟

```promql
sum by (provider, status) (
  rate(external_call_total[5m])
)
```

```promql
histogram_quantile(
  0.95,
  sum by (le, provider, operation) (
    rate(external_call_duration_seconds_bucket[5m])
  )
)
```

### 5. 业务结果层

业务结果层回答“业务是否达成目标”：

- 订单创建数
- 支付成功率
- 消息投递成功率
- 用户注册数
- 文件处理成功率
- 审核通过率
- AI 任务有效率

这些指标不一定适合分钟级告警，但非常适合做趋势分析和复盘。

## 八、告警规则设计

监控最终要服务于行动。告警规则应该满足两个条件：能反映真实风险，且不会频繁误报。

### 1. 高错误率告警

```yaml
groups:
  - name: service.rules
    rules:
      - alert: HighHttp5xxRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          /
          clamp_min(sum(rate(http_requests_total[5m])), 0.001)
          > 0.05
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "HTTP 5xx error rate is high"
```

### 2. 高延迟告警

```yaml
      - alert: HighRequestLatency
        expr: |
          histogram_quantile(
            0.95,
            sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
          ) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "HTTP request latency is high"
```

### 3. 队列堆积告警

```yaml
      - alert: QueueBacklogHigh
        expr: queue_size{queue_name="default"} > 1000
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Queue backlog is high"
```

### 4. 外部依赖失败率告警

```yaml
      - alert: ExternalDependencyFailureRateHigh
        expr: |
          sum by (provider) (rate(external_call_total{status!="success"}[5m]))
          /
          clamp_min(sum by (provider) (rate(external_call_total[5m])), 0.001)
          > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "External dependency failure rate is high"
```

告警阈值不应该只凭经验拍脑袋。更合理的做法是先观察一段时间的正常波动范围，再设置阈值，并结合业务优先级区分 warning 和 critical。

## 九、标签设计：避免高基数问题

Prometheus 指标由指标名和标签共同决定时间序列。标签越多、标签值越分散，时间序列数量越大。高基数标签是 Prometheus 使用中的常见问题。

不建议作为标签的内容：

- 用户 ID
- 请求 ID
- trace ID
- 手机号、邮箱等个人信息
- 订单号、支付单号
- 原始异常堆栈
- 完整 URL 查询参数
- 原始 SQL
- 原始消息内容

适合作为标签的内容：

- HTTP method
- 规范化 path，例如 `/orders/{id}` 而不是 `/orders/123`
- status
- service
- environment
- operation
- provider
- flow_name
- stage_name

例如，下面这个指标设计风险很高：

```text
http_requests_total{path="/orders/1029384756", user_id="u_123456"}
```

更合理的方式是：

```text
http_requests_total{path="/orders/{id}", method="GET", status="200"}
```

trace ID 应该放在日志或链路追踪系统中，而不是放在 Prometheus 标签里。Prometheus 适合看聚合趋势，日志适合还原单次细节，Trace 适合分析跨服务调用链。

## 十、从零接入的一套推荐步骤

如果一个项目从零开始接入 Prometheus 和 Grafana，可以按以下顺序推进：

1. 在应用中引入 Prometheus 客户端库。
2. 暴露 `/metrics` 端点。
3. 先记录 HTTP 请求量、状态码和耗时。
4. 为核心业务流程增加 Counter 和 Histogram。
5. 为队列长度、活跃任务数等状态增加 Gauge。
6. 为外部依赖调用增加次数、状态和耗时指标。
7. 配置 Prometheus 抓取应用指标。
8. 在 Grafana 中添加 Prometheus 数据源。
9. 先建立总览仪表盘，再补充接口、业务流程和依赖面板。
10. 观察一段时间后再设置告警阈值。
11. 在故障复盘后持续调整指标，而不是一次性设计完。

这套顺序的重点是“先可用，再完善”。初期不需要把所有指标一次性设计完整，先覆盖最关键路径，后续根据排障需要逐步补充。

## 十一、常见误区

### 误区一：只接入系统指标，不接入业务指标

CPU、内存正常不代表业务正常。一个支付系统即使 CPU 很低，也可能因为第三方回调失败导致支付状态无法更新。

### 误区二：指标越多越好

指标太多会增加维护成本，也会让仪表盘失去重点。指标应该围绕问题设计，而不是围绕代码行数设计。

### 误区三：把 Prometheus 当日志系统

Prometheus 不适合保存每次请求的详细上下文。原始请求、异常堆栈、trace ID、用户输入等信息应该进入日志或链路追踪。

### 误区四：过早设置告警

如果不了解系统正常波动范围，告警很容易误报。建议先观察基线，再设置阈值。

### 误区五：只看平均值

平均值会掩盖长尾延迟。接口平均耗时 100ms，并不代表没有用户遇到 5s 请求。生产监控中应重点关注 P95、P99。

## 总结

Prometheus 和 Grafana 的接入并不复杂：应用暴露 `/metrics`，Prometheus 定时抓取，Grafana 查询并展示。但真正决定监控质量的，不是工具本身，而是指标设计。

通用项目可以按照以下思路建立监控体系：

- 用 Counter 记录请求量、成功量、失败量。
- 用 Gauge 记录当前状态，如队列长度、活跃任务数。
- 用 Histogram 记录耗时分布，并关注 P95、P99。
- 从 HTTP 层逐步深入到业务流程层、外部依赖层和业务结果层。
- 控制标签基数，把日志、Trace 和指标的职责边界划清楚。
- 先建立可用仪表盘，再根据真实故障和复盘持续迭代。

当这套体系建立起来后，团队不只知道“服务有没有挂”，还可以知道“业务链路哪里慢了、哪里失败了、哪个依赖不稳定、哪个指标正在偏离正常范围”。这才是 Prometheus 和 Grafana 在工程实践中的真正价值。

## 参考资料

- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Prometheus Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)
- [Grafana Prometheus Data Source](https://grafana.com/docs/grafana/latest/datasources/prometheus/)
