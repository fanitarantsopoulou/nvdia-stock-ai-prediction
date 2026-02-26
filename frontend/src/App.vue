<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { Line } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement, Filler } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement, Filler)

const prediction = ref(null)
const chartData = ref(null)
const loading = ref(true)
const logs = ref([])

const addLog = (msg) => {
  logs.value.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`)
  if (logs.value.length > 8) logs.value.pop()
}

const fetchData = async () => {
  loading.value = true
  addLog("INITIALIZING NEURAL_CORE...")
  try {
    const res = await axios.get('http://127.0.0.1:8000/predict')
    prediction.value = res.data
    addLog("FETCHING MARKET_DATA: SUCCESS")
    addLog(`SENTIMENT_SCORE: ${res.data.sentiment_score}`)
    addLog("RUNNING LSTM_INFERENCE...")

    const formatTime = (dateStr) => {
      const date = new Date(dateStr);
      return date.toLocaleString('en-US', { hour: 'numeric', minute: '2-digit', hour12: false });
    }

    const historicalLabels = res.data.history_times.map(t => formatTime(t))
    const lastDate = new Date(res.data.history_times[res.data.history_times.length - 1]);
    const nextHour = new Date(lastDate.getTime() + 60 * 60 * 1000);
    const forecastLabel = formatTime(nextHour) + " >> PRED";

    chartData.value = {
      labels: [...historicalLabels, forecastLabel],
      datasets: [{
        label: 'PRICE_FLUX',
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 400);
          gradient.addColorStop(0, res.data.direction === 'UP' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)');
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          return gradient;
        },
        borderColor: res.data.direction === 'UP' ? '#00ff41' : '#ff003c',
        data: [...res.data.history, res.data.predicted_price],
        fill: true,
        tension: 0.2,
        borderWidth: 2,
        pointBackgroundColor: '#fff',
        pointRadius: (context) => context.dataIndex === res.data.history.length ? 6 : 0,
        borderDash: (context) => context.dataIndex === res.data.history.length ? [5, 5] : [],
      }]
    }
    addLog("PREDICTION_READY: " + res.data.direction)
  } catch (e) {
    addLog("ERR: CONNECTION_TERMINATED");
  } finally {
    loading.value = false
  }
}
onMounted(() => {
  fetchData();
  setInterval(fetchData, 300000); 
})

</script>

<template>
  <div class="tech-bg"></div>
  <div class="scanlines"></div>

  <div class="relative z-10 min-h-screen text-[#00ff41] font-mono p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
      
      <header class="flex justify-between items-end mb-8 border-b-2 border-[#00ff41] pb-2 bg-black bg-opacity-40 backdrop-blur-md px-4">
        <div>
          <h1 class="text-3xl font-black tracking-tighter uppercase italic">
            NVDA STOCK <span class="text-white">AI PREDICTION</span>
          </h1>
        </div>
        <div class="text-right">
          <div class="text-xs font-bold animate-pulse text-black bg-[#00ff41] px-2 mb-1 shadow-[0_0_10px_#00ff41]">SYSTEM_LIVE</div>
          <div class="text-[10px] opacity-60">* LOCAL_SYS_TIME_SYNC</div>
        </div>
      </header>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        <div class="lg:col-span-1 bg-black bg-opacity-60 border border-[#00ff41] p-4 text-[10px] h-[450px] flex flex-col shadow-[inset_0_0_15px_rgba(0,255,65,0.1)]">
          <h3 class="border-b border-[#00ff41] mb-4 pb-1 font-bold tracking-widest text-white uppercase">Terminal_Logs</h3>
          <div class="flex-1 overflow-hidden space-y-3 opacity-80">
            <div v-for="(log, i) in logs" :key="i" :class="i === 0 ? 'text-white' : 'text-[#00ff41] opacity-50'">
              {{ log }}
            </div>
          </div>
          <div class="mt-4 pt-2 border-t border-[#00ff41] text-[9px] opacity-40">
            SECURE_CONNECTION_STABLE
          </div>
        </div>

        <div class="lg:col-span-3 space-y-6">
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative group">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase">Value</span>
              <p class="text-[10px] opacity-50 uppercase">Current USD</p>
              <p class="text-3xl font-bold tracking-tighter text-white">${{ prediction?.current_price }}</p>
            </div>
            
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase italic">AI_Forecast</span>
              <p class="text-[10px] opacity-50 uppercase italic">1H_Prediction</p>
              <p :class="prediction?.direction === 'UP' ? 'text-[#00ff41]' : 'text-[#ff003c]'" class="text-3xl font-bold tracking-tighter">
                ${{ prediction?.predicted_price }}
              </p>
            </div>

            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase">Analysis</span>
              <p class="text-[10px] opacity-50 uppercase">Sentiment</p>
              <p class="text-3xl font-bold text-blue-400 tracking-tighter">{{ prediction?.sentiment_score }}</p>
            </div>
          </div>

          <div class="border border-[#00ff41] p-6 bg-black bg-opacity-60 relative h-[360px] shadow-[0_0_30px_rgba(0,255,65,0.05)]">
            <div class="flex justify-between items-center mb-6">
               <span class="text-xs font-bold tracking-[0.3em] uppercase">Market_Visualizer_V1</span>
               <button @click="fetchData" class="border border-[#00ff41] text-[10px] px-4 py-1 hover:bg-[#00ff41] hover:text-black transition-all duration-300 font-bold">
                 RUN_ANALYSIS()
               </button>
            </div>
            
            <div v-if="loading" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-90 z-20 border border-[#00ff41]">
              <p class="animate-pulse font-bold tracking-[0.5em] text-white">RECALIBRATING_NEURAL_NET...</p>
            </div>

            <div class="h-[260px]">
              <Line v-if="chartData" :data="chartData" :options="{ 
                maintainAspectRatio: false,
                scales: {
                  y: { grid: { color: 'rgba(0, 255, 65, 0.05)' }, ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10 } } },
                  x: { grid: { display: false }, ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10 } } }
                },
                plugins: { legend: { display: false } }
              }" />
            </div>
          </div>
        </div>
      </div>

      <footer class="mt-8 text-center text-[9px] opacity-30 uppercase tracking-[0.5em]">
        End_to_End Encryption Active // No Data Leak Detected
      </footer>
    </div>
  </div>
</template>

<style>
/* Full screen grid movement */
.tech-bg {
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  z-index: 0;
  background-image: 
    linear-gradient(rgba(0, 255, 65, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 65, 0.05) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: bg-move 120s linear infinite;
}

@keyframes bg-move {
  from { background-position: 0 0; }
  to { background-position: 0 1000px; }
}

/* CRT Scanline look */
.scanlines {
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  z-index: 100;
  background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.15) 50%);
  background-size: 100% 4px;
  pointer-events: none;
  opacity: 0.4;
}

/* Background darkness */
body {
  background-color: #050505;
  margin: 0;
}
</style>