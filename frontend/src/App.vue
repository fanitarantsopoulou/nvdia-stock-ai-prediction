<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { Line } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement)

const prediction = ref(null)
const chartData = ref(null)

const fetchData = async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8000/predict')
    prediction.value = res.data
    
    // Mocking some history for the chart (In real life, get this from API)
    chartData.value = {
      labels: ['-5h', '-4h', '-3h', '-2h', '-1h', 'Now', 'Forecast'],
      datasets: [{
        label: 'NVDA Price ($)',
        backgroundColor: '#10b981',
        borderColor: '#10b981',
        data: [192, 194, 193, 195, 194, res.data.current_price, res.data.predicted_price],
        stepped: false,
      }]
    }
  } catch (e) { console.error(e) }
}

onMounted(fetchData)
</script>

<template>
  <div class="min-h-screen bg-black text-white font-sans p-6">
    <div class="max-w-5xl mx-auto">
      <header class="flex justify-between items-center mb-10 border-b border-gray-800 pb-4">
        <h1 class="text-2xl font-bold tracking-tighter uppercase">Nvidia <span class="text-green-500">AI-Quant</span></h1>
        <div class="text-xs text-gray-500 font-mono">2026-02-26 | LIVE FEED</div>
      </header>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p class="text-gray-500 text-sm mb-1">Current Price</p>
          <p class="text-4xl font-mono">${{ prediction?.current_price }}</p>
        </div>

        <div class="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p class="text-gray-500 text-sm mb-1">AI Forecast (1H)</p>
          <p :class="prediction?.direction === 'UP' ? 'text-green-500' : 'text-red-500'" class="text-4xl font-mono">
            ${{ prediction?.predicted_price }}
          </p>
        </div>

        <div class="bg-gray-900 p-6 rounded-lg border border-gray-800">
          <p class="text-gray-500 text-sm mb-1">Sentiment Score</p>
          <p class="text-4xl font-mono text-blue-400">{{ prediction?.sentiment_score }}</p>
        </div>
      </div>

      <div class="mt-8 bg-gray-900 p-6 rounded-lg border border-gray-800 h-96">
        <Line v-if="chartData" :data="chartData" :options="{ maintainAspectRatio: false }" />
      </div>
    </div>
  </div>
</template>