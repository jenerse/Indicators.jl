using Indicators
using JSON

include("./300001.jl")

stock_data = JSON.parse(stock300001)
n = length(stock_data)

ohlcv = zeros(Float64, n, 5)

for i in 1:n
    target = stock_data[i]
    ohlcv[i, :] = [target["open"], target["high"], target["low"], target["close"], target["volume"]]
end

result = Indicators.dkx(@views ohlcv[:, 1:4])
