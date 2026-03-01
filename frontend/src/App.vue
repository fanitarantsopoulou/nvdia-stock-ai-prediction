<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Filler,
} from 'chart.js'

// ---------------------------------------------------------------------------
// Chart.js registration
// ---------------------------------------------------------------------------
ChartJS.register(
  Title, Tooltip, Legend,
  LineElement, CategoryScale, LinearScale,
  PointElement, Filler,
)

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const prediction   = ref(null)
const chartData    = ref(null)
const loading      = ref(true)
const logs         = ref([])
const isStalePrice = ref(false)  // true when backend signals price_source === 'stale'

let activeController = null

// Polling interval handle — stored at module scope so onUnmounted can clear it.
let pollingInterval = null

// ---------------------------------------------------------------------------
// Market status badge config
// Covers all statuses returned by the (updated) predict.py, including
// CLOSED_HOLIDAY which the original switch was missing.
// ---------------------------------------------------------------------------
const marketStatusConfig = computed(() => {
  if (!prediction.value) {
    return { text: 'AWAITING_TELEMETRY', class: 'bg-gray-600 text-white' }
  }

  switch (prediction.value.market_status) {
    case 'MARKET_OPEN':
      return {
        text: 'MARKET_OPEN',
        class: 'bg-[#00ff41] text-black shadow-[0_0_10px_#00ff41] animate-pulse',
      }
    case 'PRE_MARKET':
      return {
        text: 'PRE_MARKET',
        class: 'bg-yellow-400 text-black shadow-[0_0_10px_rgba(250,204,21,0.5)]',
      }
    case 'AFTER_HOURS':
      return {
        text: 'AFTER_HOURS',
        class: 'bg-blue-500 text-white shadow-[0_0_10px_rgba(59,130,246,0.5)]',
      }
    case 'CLOSED_WEEKEND':
      return {
        text: 'CLOSED_WEEKEND',
        class: 'bg-red-600 text-white shadow-[0_0_10px_rgba(220,38,38,0.5)]',
      }
    case 'CLOSED_HOLIDAY':
      return {
        text: 'CLOSED_HOLIDAY',
        class: 'bg-orange-500 text-white shadow-[0_0_10px_rgba(249,115,22,0.5)]',
      }
    case 'CLOSED':
      return {
        text: 'MARKET_CLOSED',
        class: 'bg-red-600 text-white shadow-[0_0_10px_rgba(220,38,38,0.5)]',
      }
    default:
      return { text: 'UNKNOWN', class: 'bg-gray-600 text-white' }
  }
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400 },
  scales: {
    y: {
      grid:  { color: 'rgba(0, 255, 65, 0.05)' },
      ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10, family: 'monospace' } },
    },
    x: {
      grid:  { display: false },
      ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10, family: 'monospace' } },
    },
  },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#000',
      borderColor:     '#00ff41',
      borderWidth:     1,
      titleFont:       { family: 'monospace' },
      bodyFont:        { family: 'monospace' },
    },
  },
}))

// ---------------------------------------------------------------------------
// Terminal log helper
// ---------------------------------------------------------------------------
const addLog = (msg) => {
  logs.value.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`)
  if (logs.value.length > 8) logs.value.pop()
}

// ---------------------------------------------------------------------------
// Core fetch function
// ---------------------------------------------------------------------------
const fetchData = async () => {

  if (activeController) {
    activeController.abort()
  }
  activeController = new AbortController()

  loading.value = true
  addLog('INITIALIZING NEURAL_CORE...')

  try {
    const res = await axios.get(`${API_BASE}/predict`, {
      signal: activeController.signal,
    })

    // Guard: backend error payload
    if (res.data?.status === 'error') {
      throw new Error(res.data.message ?? 'Backend returned error status')
    }

    // Guard: missing or empty history
    if (!res.data?.history_times || res.data.history_times.length === 0) {
      throw new Error('Empty data received from API')
    }

    prediction.value   = res.data
    isStalePrice.value = res.data.price_source === 'stale'

    addLog('FETCHING MARKET_DATA: SUCCESS')
    addLog(`MARKET_STATE: ${res.data.market_status}`)
    if (isStalePrice.value) {
      addLog('WARN: LIVE_PRICE_UNAVAILABLE — USING_LAST_CLOSE')
    }
    addLog('RUNNING LSTM_INFERENCE...')

    const historicalCount = res.data.history.length
    const forecastLabel   = `PRED >> ${res.data.forecast_time}`
    const accentColor     = res.data.direction === 'UP' ? '#00ff41' : '#ff003c'
    const accentColorBg   = res.data.direction === 'UP'
      ? 'rgba(0, 255, 101, 0.3)'
      : 'rgba(255, 0, 60, 0.3)'

    chartData.value = {
      labels: [...res.data.history_times, forecastLabel],
      datasets: [
        {
          // Historical price line
          label:            'HISTORICAL',
          data:             res.data.history,
          borderColor:      accentColor,
          borderWidth:      2,
          tension:          0.3,
          fill:             false,
          pointRadius:      0,
          pointHoverRadius: 6,
          pointBackgroundColor: '#fff',
          backgroundColor:  accentColorBg,
        },
        {
          // Prediction bridge — starts at the last historical point so the
          // chart line is visually continuous with no gap.
          label:       'AI_PREDICTION',
          data: [
            // Pad with nulls so this dataset starts exactly at the last historical index
            ...Array(historicalCount - 1).fill(null),
            res.data.history[historicalCount - 1], // bridge anchor (last real price)
            res.data.predicted_price,
          ],
          borderColor:      accentColor,
          borderWidth:      2,
          borderDash:       [5, 5],
          tension:          0.3,
          fill:             false,
          pointRadius:      (context) =>
            context.dataIndex === historicalCount ? 6 : 0,
          pointHoverRadius: 8,
          pointBackgroundColor: '#fff',
        },
      ],
    }

    addLog(`PREDICTION_READY: ${res.data.direction} @ $${res.data.predicted_price}`)

  } catch (err) {
    // axios throws a CanceledError when we call controller.abort() — this is
    // intentional and should not be treated as a real error.
    if (axios.isCancel(err)) {
      addLog('INFO: PREVIOUS_REQUEST_SUPERSEDED')
      return
    }

    console.error('Critical API Error:', err)
    addLog(`ERR: ${err.message ?? 'CONNECTION_TERMINATED'}`)
    chartData.value    = null
    isStalePrice.value = false
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchData()
  pollingInterval = setInterval(fetchData, 5 * 60 * 1000) // 5 minutes
})

onUnmounted(() => {
  // Stop polling
  if (pollingInterval !== null) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
  // Cancel any pending HTTP request so it doesn't resolve into a dead component
  if (activeController) {
    activeController.abort()
    activeController = null
  }
})
</script>

<template>
  <div class="tech-bg"></div>
  <div class="scanlines"></div>

  <div class="relative z-10 min-h-screen text-[#00ff41] font-mono p-4 md:p-8">
    <div class="max-w-6xl mx-auto">

      <!-- ================================================================
           Header
           ================================================================ -->
      <header class="flex justify-between items-end mb-8 border-b-2 border-[#00ff41] pb-2 bg-black bg-opacity-40 backdrop-blur-md px-4">
        <div>
          <h1 class="text-3xl font-black tracking-tighter uppercase italic">
            NVDA STOCK <span class="text-white">AI PREDICTION</span>
          </h1>
          <p class="text-[9px] opacity-50 tracking-widest uppercase mt-1">Nasdaq Real-Time Feed Active</p>
        </div>

        <div class="text-right flex flex-col items-end gap-1">
          <!-- Market status badge -->
          <div :class="['text-xs font-bold px-2 tracking-wider', marketStatusConfig.class]">
            {{ marketStatusConfig.text }}
          </div>

          <div
            v-if="isStalePrice"
            class="text-[9px] px-2 py-0.5 bg-orange-500 text-black font-bold uppercase tracking-wider animate-pulse"
          >
            ⚠ STALE_PRICE — LIVE_FEED_DOWN
          </div>

          <div class="text-[10px] opacity-60">* NASDAQ_TZ_SYNC_ENABLED</div>
        </div>
      </header>

      <!-- ================================================================
           Main grid
           ================================================================ -->
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">

        <!-- Terminal log panel -->
        <div class="lg:col-span-1 bg-black bg-opacity-60 border border-[#00ff41] p-4 text-[10px] h-[450px] flex flex-col shadow-[inset_0_0_15px_rgba(0,255,65,0.1)]">
          <h3 class="border-b border-[#00ff41] mb-4 pb-1 font-bold tracking-widest text-white uppercase">
            Terminal_Logs
          </h3>
          <div class="flex-1 overflow-y-auto space-y-3 opacity-80 custom-scrollbar">
            <div
              v-for="(log, i) in logs"
              :key="i"
              :class="i === 0 ? 'text-white font-bold' : 'text-[#00ff41] opacity-50'"
            >
              {{ log }}
            </div>
          </div>
          <div class="mt-4 pt-2 border-t border-[#00ff41] text-[9px] opacity-40">
            STABLE_DATALINK_ESTABLISHED
          </div>
        </div>

        <!-- Right column -->
        <div class="lg:col-span-3 space-y-6">

          <!-- KPI cards -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">

            <!-- Current price -->
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase">
                Live
              </span>
              <p class="text-[10px] opacity-50 uppercase font-bold">Current USD</p>
              <p class="text-3xl font-bold tracking-tighter text-white">
                ${{ prediction?.current_price ?? '---' }}
              </p>
              <!-- Stale indicator on the card itself for extra clarity -->
              <p v-if="isStalePrice" class="text-[8px] text-orange-400 mt-1 uppercase">
                ⚠ last close (live feed unavailable)
              </p>
            </div>

            <!-- Predicted price -->
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase italic">
                AI_Inference
              </span>
              <p class="text-[10px] opacity-50 uppercase italic font-bold">
                {{ prediction?.market_status === 'MARKET_OPEN' ? '1H_Forecast' : 'Next_Open_Forecast' }}
              </p>
              <p
                :class="prediction?.direction === 'UP' ? 'text-[#00ff41]' : 'text-[#ff003c]'"
                class="text-3xl font-bold tracking-tighter"
              >
                ${{ prediction?.predicted_price ?? '---' }}
              </p>
            </div>

            <!-- Sentiment score -->
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-blue-500 text-white font-bold uppercase font-mono">
                FinBert
              </span>
              <p class="text-[10px] opacity-50 uppercase font-bold">Sentiment_Index</p>
              <p class="text-3xl font-bold text-blue-400 tracking-tighter">
                {{ prediction?.sentiment_score ?? '---' }}
              </p>
            </div>
          </div>

          <!-- Chart panel -->
          <div class="border border-[#00ff41] p-6 bg-black bg-opacity-60 relative h-[360px] shadow-[0_0_30px_rgba(0,255,65,0.05)]">
            <div class="flex justify-between items-center mb-6">
              <span class="text-xs font-bold tracking-[0.3em] uppercase">
                Market_Visualizer_V1
              </span>
              <button
                @click="fetchData"
                :disabled="loading"
                class="border border-[#00ff41] text-[10px] px-4 py-1 font-bold transition-all duration-300 active:scale-95"
                :class="loading
                  ? 'opacity-40 cursor-not-allowed'
                  : 'hover:bg-[#00ff41] hover:text-black cursor-pointer'"
              >
                {{ loading ? 'PROCESSING...' : 'RUN_ANALYSIS()' }}
              </button>
            </div>

            <!-- Loading overlay -->
            <div
              v-if="loading"
              class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-90 z-20 border border-[#00ff41]"
            >
              <p class="animate-pulse font-bold tracking-[0.5em] text-white uppercase">
                Recalibrating_Neural_Weights...
              </p>
            </div>

            <!-- Chart -->
            <div class="h-[260px]">
              <Line
                v-if="chartData"
                :data="chartData"
                :options="chartOptions"
              />

              <!-- Empty state when API has failed -->
              <div
                v-else-if="!loading"
                class="h-full flex items-center justify-center text-[#ff003c] text-xs tracking-widest uppercase"
              >
                DATA_FEED_OFFLINE — RETRYING_IN_5M
              </div>
            </div>
          </div>

        </div>
      </div>

      <!-- ================================================================
           Footer
           ================================================================ -->
      <footer class="mt-8 text-center text-[9px] opacity-30 uppercase tracking-[0.5em]">
        Neural Architecture v1.0 // No Data Leak Detected // NVIDIA_QUANTA_CORE
      </footer>

    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.3);
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #00ff41;
}
</style>