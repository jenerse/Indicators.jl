# Momentum-oriented technical indicator functions
"""
```
llv(input::Array{T}; n::Int)::Array{T}
```

find min in range of an array and return array

*OutPut*

- Array{T}
"""
function llv(input::AbstractArray{T}, period::Int64)::Array{T} where {T<:Real}
    n = length(input)
    out = Array{Float64}(undef, n)
    
    @inbounds for i in 1:n
        if i < period
            out[i] = minimum(@views input[1:i])
        else
            out[i] = minimum(@views input[(i-period+1):i])
        end
    end
    
    return out
end

"""
```
hhv(input::Array{T}; n::Int)::Array{T}
```

find max in range of an array and return array

*OutPut*

- Array{T}
"""
function hhv(input::AbstractArray{T}, period::Int64)::Array{T} where {T<:Real}
    n = length(input)
    out = Array{Float64}(undef, n)
    
    @inbounds for i in 1:n
        if i < period
            out[i] = maximum(@views input[1:i])
        else
            out[i] = maximum(@views input[(i-period+1):i])
        end
    end
    
    return out
end

"""
```
suma(input::AbstractArray{T}, n::Int)::Array{Float64}
```

sum array by range and return array

*OutPut*

- Array{Float64}
"""
function suma(x::AbstractArray{T}, step::Int64)::Array{T} where {T<:Real}
    n = length(x)
    result = Array{Float64}(undef, n)
    
    @inbounds for i in 1:n
        if i <= step
            result[i] = sum(@views x[1:i])
        else
            result[i] = sum(@views x[(i-step+1):i])
        end
    end
    
    return result
end

"""
```
bbi(x::AbstractArray{T}; m1::Int64=3, m2::Int64=6, m3::Int64=12, m4::Int64=24)::Array{Float64}
```

bbi

*Output*

Array{Float64}
"""
function bbi(x::AbstractArray{T}; m1::Int64=3, m2::Int64=6, m3::Int64=12, m4::Int64=24)::Array{Float64} where {T<:Real}
    len = length(x)
    ma1 = len >= m1 ? sma(x; n=m1) : fill(0, len)
    ma2 = len >= m2 ? sma(x; n=m2) : fill(0, len)
    ma3 = len >= m3 ? sma(x; n=m3) : fill(0, len)
    ma4 = len >= m4 ? sma(x; n=m4) : fill(0, len)
    out = (ma1 + ma2 + ma3 + ma4) / 4
    out[1:min(len, max(m1, m2, m3, m4))] .= NaN
    return out
end

"""
```
skdj(hlc::AbstractMatrix{T}; m1::Int64=9, m2::Int64=3)::Matrix{Float64}
```

skdj k/d

*Output*

- Column 1: k
- Column 2: d
"""
function skdj(hlc::AbstractMatrix{T}; m1::Int64=9, m2::Int64=3)::Matrix{Float64} where {T<:Real}
    len = size(hlc, 1)
    out = zeros(Float64, len, 2)
    lowv = llv(hlc[:, 2], m1)
    highv = hhv(hlc[:, 1], m1)

    rsv = (hlc[:, 3] .- lowv) ./ (highv .- lowv) * 100
    rsv = ifelse.(isinf.(rsv), 0.0, rsv)

    rsv = emadc(rsv; n=m2)
    out[:, 1] = emadc(rsv; n=m2)
    out[:, 2] = sma(out[:, 1]; n=m2)

    return out
end

"""
```
kdj(hlc::AbstractMatrix{T}; m1::Int64=9, m2::Int64=6, m3::Int64=3)::Matrix{Float64}
```

kdj k/d/j

*Output*

- Column 1: k
- Column 2: d
- Column 3: j
"""
function kdj(hlc::AbstractMatrix{T}; m1::Int64=9, m2::Int64=6, m3::Int64=3)::Matrix{Float64} where {T<:Real}
    len = size(hlc, 1)
    out = zeros(Float64, len, 3)
    lowv = llv(hlc[:, 2], m1)
    highv = hhv(hlc[:, 1], m1)

    rsv = (hlc[:, 3] .- lowv) ./ (highv .- lowv) * 100
    rsv = ifelse.(isinf.(rsv), 0.0, rsv)

    out[:, 1] = smadc(rsv, m2, 1; fillnan=false)
    out[m2:end, 2] = smadc(out[m2:end, 1], m3, 1; fillnan=false)
    out[:, 3] = 3 * out[:, 1] - 2 * out[:, 2]

    out[1:m2-1, :] .= NaN

    return out
end


"""
```
dkx(ohlc::AbstractMatrix{T}; period::Int64=43)::Matrix{Float64}
```

dkx dkx/madkx

*Output*

- Column 1: dkx
- Column 2: madkx
"""
function dkx(ohlc::AbstractMatrix{T}; period::Int64=43)::Matrix{Float64} where {T<:Real}
    @assert size(ohlc, 2) == 4 "Argument `ohlc` must have exactly 4 columns."
    n = size(ohlc, 1)
    out = zeros(Float64, n, 2)

    @inbounds for i in n:-1:1
        if i >= 21
            mid = 0.0
            @inbounds for _i in 0:19
                targetIndex = _i == 19 ? i - 20 : i - _i
                mid += (20 - _i) * (sum(ohlc[targetIndex, 1:3]) + ohlc[targetIndex, 4] * 3) / 6
                out[i, 1] = mid / 210
            end
        end
    end
    out[:, 2] = n >= period ? sma(out[:, 1]; n=period) : fill(NaN, n)
    out[1:min(20, n), :] .= NaN

    return out
end

"""
```
lon(hlcv::AbstractMatrix{T}; period::Int64=43)::Matrix{Float64}
```

lon lon/lonma

*Output*

- Column 1: lon
- Column 2: lonma
"""
function lon(hlcv::AbstractMatrix{T}; period::Int64=43)::Matrix{Float64} where {T<:Real}
    @assert size(hlcv, 2) == 4 "Argument `hlcv` must have exactly 2 columns."
    len = size(hlcv, 1)
    out = zeros(T, len, 2)

    volSum = suma(hlcv[:, 4], 2)
    lowv = llv(hlcv[:, 2], 2)
    highv = hhv(hlcv[:, 1], 2)

    lc = @views hlcv[1:end-1, 3]
    vids = @views volSum[2:end] ./ (( @views highv[2:end] -  @views lowv[2:end]) * 100)
    rc = (@views hlcv[2:end, 3] - lc) .* vids
    pushfirst!(rc, 0)

    long = cumsum(rc)
    diff = smadc(long, 10, 1; fillnan=false)
    dea = smadc(long, 20, 1; fillnan=false)

    out[:, 1] = diff - dea
    out[:, 2] = len >= period ? sma(out[:, 1]; n=period) : fill(NaN, len)

    return out
end

"""
```
obv(cv::Matrix{T}; n::Int64=37)::Matrix{Float64}
```

Obv obv/maobv

*Output*

- Column 1: obv
- Column 2: maobv
"""
function obv(cv::AbstractMatrix{T}; period::Int64=37)::Matrix{Float64} where {T<:Real}
    @assert size(cv, 2) == 2 "Argument `cv` must have exactly 2 columns."
    n = size(cv, 1)
    out = zeros(T, n, 2)

    out[1, 1] = 0

    @inbounds for i in 2:n
        item = cv[i, :]
        prevItem = cv[i-1, :]

        if item[1] > prevItem[1] # close > prevClose
            out[i, 1] = out[i-1, 1] + item[2]
        elseif item[1] < prevItem[1] # close < prevClose
            out[i, 1] = out[i-1, 1] - item[2]
        else
            out[i, 1] = out[i-1, 1]
        end
    end
    out[:,2] = length(out[:,1]) >= period ? sma(out[:,1]; n=period) : fill(NaN, n)
    return out
end

"""
```
aroon(hl::Matrix{T}; n::Int64=25)::Matrix{Float64}
```

Aroon up/down/oscillator

*Output*

- Column 1: Aroon Up
- Column 2: Aroon Down
- Column 3: Aroon Oscillator
"""
function aroon(hl::AbstractMatrix{T}; n::Int64=25)::Matrix{Float64} where {T<:Real}
    @assert size(hl,2) == 2 "Argument `hl` must have exactly 2 columns."
    @assert n < size(hl,1) "Argument `n` must be less than the number of rows in argument `hl`."
    out = zeros(T, (size(hl,1),3))
    @inbounds for i in n:size(hl,1)
        out[i,1] = 100.0 * (n - findmax(hl[i-n+1:i,1])[2]) / 25.0
    end
    @inbounds for i in n:size(hl,1)
        out[i,2] = 100.0 * (n - findmin(hl[i-n+1:i,2])[2]) / 25.0
    end
    out[:,3] = out[:,1]-out[:,2]  # aroon oscillator
    out[1:n-1,:] .= NaN
    return out
end

"""
```
donch(hl::Matrix{T}; n::Int64=10, inclusive::Bool=true)::Array{Float64}
```

Donchian channel (if inclusive is set to true, will include current bar in calculations.)

*Output*

- Column 1: Lowest low of last `n` periods
- Column 2: Average of highest high and lowest low of last `n` periods
- Column 3: Highest high of last `n` periods
"""
function donch(hl::AbstractMatrix{T}; n::Int64=10, inclusive::Bool=true)::Matrix{Float64} where {T<:Real}
    @assert size(hl,2) == 2 "Argument `hl` must have exactly 2 columns."
    local lower::Array{T} = runmin(hl[:,2], n=n, cumulative=false, inclusive=inclusive)
    local upper::Array{T} = runmax(hl[:,1], n=n, cumulative=false, inclusive=inclusive)
    local middle::Array{T} = (lower .+ upper) ./ 2.0
    return [lower middle upper]
end

function shft(x::AbstractVector{T} where T<:Number, n::Integer)
    if n == 0
        return x
    elseif n < 0
        return vcat(x[1-n:end], fill(NaN, -n))
    else
        return vcat(fill(NaN, n), x[1:end-n])
    end
end

"""
```
ichimoku(hlc::AbstractMatrix{T}; params=(9, 26, 26, 52, -26))::Matrix{Float64} where {T<:Real}
```

Ichimoku Kinko Hyo

*Output*

- Column 1: Tenkan-sen
- Column 2: Kijun-sen
- Column 3: Senkou Span A
- Column 4: Senkou Span B
- Column 5: Chikou Span
"""
function ichimoku(hlc::AbstractMatrix{T}; params=(10, 26, 26, 52, -26))::Matrix{Float64} where {T<:Real}
    # Source: https://www.investopedia.com/terms/i/ichimokuchart.asp
    # TODO: Implement option to get forward looking Senkous
    @assert size(hlc,2) == 3 "Argument `hlc` must have exactly 3 columns."
    tenkan = donch(hlc[:,1:2]; n=params[1], inclusive=true)[:,2]
    kijun = donch(hlc[:,1:2]; n=params[2], inclusive=true)[:,2]
    senkoua = shft((tenkan + kijun) /2, params[3])
    senkoub = shft(donch(hlc[:,1:2]; n=params[4], inclusive=true)[:,2], params[3])
    chikou = shft(hlc[:,3], params[5])
    return [tenkan kijun senkoua senkoub chikou]
end

"""
```
momentum(x::Array{T}; n::Int64=1)::Array{Float64}
```

Momentum indicator (price now vs price `n` periods back)
"""
function momentum(x::AbstractArray{T}; n::Int64=1)::Array{Float64} where {T<:Real}
    @assert n>0 "Argument n must be positive."
    return diffn(x, n=n)
end

"""
```
roc(x::Array{T}; n::Int64=1)::Array{Float64}
```

Rate of change indicator (percent change between i'th observation and (i-n)'th observation)
"""
function roc(x::AbstractArray{T}; n::Int64=1)::Array{Float64} where {T<:Real}
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    out = zeros(size(x)) .* NaN
    @inbounds for i = (n+1):size(x,1)
        out[i] = x[i]/x[i-n] - 1.0
    end
    return out * 100.0
end

"""
```
macdAgu(x::AbstractArray{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9)::Array{Float64}
```

A股计算macd方法

*Output*

- Column 1: MACD (DIF)
- Column 2: MACD Signal Line (DEA)
- Column 3: MACD Histogram (MACD柱)
"""
function macdAgu(x::AbstractArray{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9)::Matrix{Float64} where {T<:Real}
    n = length(x)
    out = zeros(T, (n,3))
    out[:, 1] = emadc(x; n=nfast) - emadc(x; n=nslow)
    out[:, 2] = emadc(out[:, 1]; n=nsig)
    out[:, 3] = (out[:,1] - out[:,2]) * 2
    return out
end

"""
```
macd(x::Array{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9)::Array{Float64}
```

Moving average convergence-divergence

*Output*

- Column 1: MACD (DIF)
- Column 2: MACD Signal Line (DEA)
- Column 3: MACD Histogram (MACD柱)
"""
function macd(x::AbstractArray{T}; nfast::Int64=12, nslow::Int64=26, nsig::Int64=9,
    fastMA::Function=ema, slowMA::Function=ema, signalMA::Function=ema)::Matrix{Float64} where {T<:Real}
    out = zeros(T, (length(x),3))
    out[:,1] = fastMA(x, n=nfast) - slowMA(x, n=nslow)
    out[:,2] = signalMA(out[:,1], n=nsig)
    out[:,3] = (out[:,1] - out[:,2]) * 2
    return out
end

"""
```
rsi(x::Array{T}; n::Int64=14, ma::Function=ema, args...)::Array{Float64}
```

Relative strength index
"""
function rsi(x::AbstractArray{T}; n::Int64=14, ma::Function=ema, args...)::Array{Float64} where {T<:Real}
    @assert n<size(x,1) && n>0 "Argument n is out of bounds."
    N = size(x,1)
    ups = zeros(N)
    dns = zeros(N)
    zro = 0.0
    dx = [NaN; ndims(x) > 1 ? diff(x, dims=1) : diff(x)]
    @inbounds for i=2:N
        if dx[i] > zro
            ups[i] = dx[i]
        elseif dx[i] < zro
            dns[i] = -dx[i]
        end
    end
    rs = [NaN; ma(ups[2:end], n=n; args...) ./ ma(dns[2:end], n=n; args...)]
    return 100.0 .- 100.0 ./ (1.0 .+ rs)
end

"""
```
adx(hlc::Array{T}; n::Int64=14, wilder=true)::Array{Float64}
```

Average directional index

*Output*

- Column 1: DI+
- Column 2: DI-
- Column 3: ADX
"""
function adx(hlc::AbstractArray{T}; n::Int64=14, ma::Function=ema, args...)::Matrix{Float64} where {T<:Real}
    @assert n<size(hlc,1) && n>0 "Argument n is out of bounds."
    if size(hlc,2) != 3
        error("HLC array must have three columns")
    end
    N = size(hlc,1)
    updm = zeros(N)
    dndm = zeros(N)
    updm[1] = dndm[1] = NaN
    @inbounds for i = 2:N
        upmove = hlc[i,1] - hlc[i-1,1]
        dnmove = hlc[i-1,2] - hlc[i,2]
        if upmove > dnmove && upmove > 0.0
            updm[i] = upmove
        elseif dnmove > upmove && dnmove > 0.0
            dndm[i] = dnmove
        end
    end
    dip = [NaN; ma(updm[2:N], n=n; args...)] ./ atr(hlc, n=n) * 100.0
    dim = [NaN; ma(dndm[2:N], n=n; args...)] ./ atr(hlc, n=n) * 100.0
    dmx = abs.(dip-dim) ./ (dip+dim)
    adx = [fill(NaN,n); ma(dmx[n+1:N], n=n; args...)] * 100.0
    return [dip dim adx]
end

"""
```
heikinashi(ohlc::AbstractMatrix{<:Real})::Matrix{Float64}
```

Heikin Ashi

*Output*

- Column 1: Heikin Ashi open -- previous (o+c)/2
- Column 2: Heikin Ashi high -- max(o,h)
- Column 3: Heikin Ashi low -- min(o,l)
- Column 4: Heikin Ashi close -- (o+h+l+c)/4
"""
function heikinashi(ohlc::AbstractMatrix{<:Real})::Matrix{Float64}
    @assert size(ohlc,2) == 4 "Argument `ohlc` must have exactly 4 columns."
    hao = vcat(NaN, mean(ohlc[1:end-1,[1,4]]; dims=2))
    hah = max.(hao, ohlc[:,2])
    hal = min.(hao, ohlc[:,3])
    hac = mean(ohlc; dims=2)
    return [hao hah hal hac]
end


"""
```
psar(hl::Array{T}; af_min::T=0.02, af_max::T=0.2, af_inc::T=af_min)::Array{Float64}
```

Parabolic stop and reverse (SAR)

*Arguments*
- `hl`: 2D array of high and low prices in first and second columns respectively
- `af_min`: starting/initial value for acceleration factor
- `af_max`: maximum acceleration factor (accel factor capped at this value)
- `af_inc`: increment to the acceleration factor (speed of increase in accel factor)
"""
function psar(hl::AbstractArray{T}; af_min::T=0.02, af_max::T=0.2, af_inc::T=af_min)::Array{Float64} where {T<:Real}
    @assert af_min<1.0 && af_min>0.0 "Argument af_min must be in [0,1]."
    @assert af_max<1.0 && af_max>0.0 "Argument af_max must be in [0,1]."
    @assert af_inc<1.0 && af_inc>0.0 "Argument af_inc must be in [0,1]."
    @assert size(hl,2) == 2 "Argument hl must have 2 columns."
    ls0 = 1
    ls = 0
    af0 = af_min
    af = 0.0
    ep0 = hl[1,1]
    ep = 0.0
    maxi = 0.0
    mini = 0.0
    sar = zeros(T,size(hl,1))
    sar[1] = hl[1,2] - std(hl[:,1]-hl[:,2])
    @inbounds for i = 2:size(hl,1)
        ls = ls0
        ep = ep0
        af = af0
        mini = min(hl[i-1,2], hl[i,2])
        maxi = max(hl[i-1,1], hl[i,1])
        # Long/short signals and local extrema
        if (ls == 1)
            ls0 = hl[i,2] > sar[i-1] ? 1 : -1
            ep0 = max(maxi, ep)
        else
            ls0 = hl[i,1] < sar[i-1] ? -1 : 1
            ep0 = min(mini, ep)
        end
        # Acceleration vector
        if ls0 == ls  # no signal change
            sar[i] = sar[i-1] + af*(ep-sar[i-1])
            af0 = (af == af_max) ? af_max : (af + af_inc)
            if ls0 == 1  # current long signal
                af0 = (ep0 > ep) ? af0 : af
                sar[i] = min(sar[i], mini)
            else  # current short signal
                af0 = (ep0 < ep) ? af0 : af
                sar[i] = max(sar[i], maxi)
            end
        else  # new signal
            af0 = af_min
            sar[i] = ep0
        end
    end
    return sar
end

"""
```
kst(x::Array{T};
    nroc::Array{Int64}=[10,15,20,30], navg::Array{Int64}=[10,10,10,15],
    wgts::Array{Int64}=collect(1:length(nroc)), ma::Function=sma)::Array{Float64}

```

KST (Know Sure Thing) -- smoothed and summed rates of change
"""
function kst(x::AbstractArray{T}; nroc::AbstractArray{Int64}=[10,15,20,30], navg::AbstractArray{Int64}=[10,10,10,15],
    wgts::AbstractArray{Int64}=collect(1:length(nroc)), ma::Function=sma)::Array{Float64} where {T<:Real}
    @assert length(nroc) == length(navg)
    @assert all(nroc.>0) && all(nroc.<size(x,1))
    @assert all(navg.>0) && all(navg.<size(x,1))
    N = length(x)
    k = length(nroc)
    out = zeros(size(x))
    @inbounds for j = 1:k
        out += ma(roc(x, n=nroc[j]), n=navg[j]) * wgts[j]
    end
    return out
end

"""
```
wpr(hlc::Matrix{T}, n::Int64=14)::Array{Float64}
```

Williams %R
"""
function wpr(hlc::AbstractMatrix{T}; n::Int64=14)::Array{Float64} where {T<:Real}
    hihi = runmax(hlc[:,1], n=n, cumulative=false)
    lolo = runmin(hlc[:,2], n=n, cumulative=false)
    return -100 * (hihi - hlc[:,3]) ./ (hihi - lolo)
end

"""
```
cci(hlc::Matrix{T}; n::Int64=20, c::T=0.015, ma::Function=sma)::Array{Float64}
```

Commodity channel index
"""
function cci(hlc::AbstractMatrix{T}; n::Int64=20, c::T=0.015, ma::Function=sma, args...)::Array{Float64} where {T<:Real}
    len = size(hlc, 1)
    if len < n
        return fill(NaN, len)
    end

    tp = (hlc[:,1] + hlc[:,2] + hlc[:,3]) ./ 3.0
    dev = runmad(tp, n=n, cumulative=false, fun=mean)
    avg = ma(tp, n=n; args...)
    return (tp - avg) ./ (c * dev)
end

"""
```
stoch(hlc::Matrix{T}; nK::Int64=14, nD::Int64=3, kind::Symbol=:fast, ma::Function=sma, args...)::Matrix{Float64}
```

Stochastic oscillator (fast or slow)
"""
function stoch(hlc::AbstractMatrix{T}; nK::Int64=14, nD::Int64=3,
    kind::Symbol=:fast, ma::Function=sma, args...)::Matrix{Float64} where {T<:Real}
    @assert kind == :fast || kind == :slow "Argument `kind` must be either :fast or :slow"
    @assert nK<size(hlc,1) && nK>0 "Argument `nK` out of bounds."
    @assert nD<size(hlc,1) && nD>0 "Argument `nD` out of bounds."
    hihi = runmax(hlc[:,1], n=nK, cumulative=false)
    lolo = runmin(hlc[:,2], n=nK, cumulative=false)
    out = zeros(T, (size(hlc,1),2))
    out[:,1] = (hlc[:,3]-lolo) ./ (hihi-lolo) * 100.0
    out[:,2] = ma(out[:,1], n=nD; args...)
    if kind == :slow
        out[:,1] = out[:,2]
        out[:,2] = ma(out[:,1], n=nD; args...)
    end
    return out
end

"""
```
smi(hlc::Matrix{T}; n::Int64=13, nFast::Int64=2, nSlow::Int64=25, nSig::Int64=9,
    maFast::Function=ema, maSlow::Function=ema, maSig::Function=sma)::Matrix{Float64}
```

SMI (stochastic momentum oscillator)
"""
function smi(hlc::AbstractMatrix{T}; n::Int64=13, nFast::Int64=2, nSlow::Int64=25, nSig::Int64=9,
    maFast::Function=ema, maSlow::Function=ema, maSig::Function=sma)::Matrix{Float64} where {T<:Real}
    hihi = runmax(hlc[:,1], n=n, cumulative=false)
    lolo = runmin(hlc[:,2], n=n, cumulative=false)
    hldif = hihi-lolo
    delta = hlc[:,3] - (hihi+lolo) / 2.0
    numer = maSlow(maFast(delta, n=nFast), n=nSlow)
    denom = maSlow(maFast(hldif, n=nFast), n=nSlow) / 2.0
    out = zeros(T, (size(hlc,1),2))
    out[:,1] = 100.0*(numer./denom)
    out[:,2] = maSig(out[:,1], n=nSig)
    return out
end
