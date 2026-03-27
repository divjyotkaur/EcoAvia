document.addEventListener('DOMContentLoaded', async () => {
    // Colors mapped to the CSS Variables
    const style = getComputedStyle(document.body);
    const cb_sarimax = '#F59E0B';
    const cb_xgboost = '#10B981';
    const cb_lstm = '#8B5CF6';
    const cb_primary = '#3B82F6';

    // Global Chart.js styling
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.elements.point.radius = 4;
    Chart.defaults.elements.point.hoverRadius = 6;
    Chart.defaults.elements.line.tension = 0.4;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
    Chart.defaults.plugins.tooltip.titleColor = '#fff';
    Chart.defaults.plugins.tooltip.padding = 12;

    // View Navigation Logic
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');
    const globalSubtitle = document.getElementById('global-subtitle');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            // Remove active classes
            navItems.forEach(nav => nav.classList.remove('active'));
            viewSections.forEach(section => section.classList.remove('active'));

            // Add active class to clicked
            item.classList.add('active');

            // Show target section
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Update subtitle appropriately
            globalSubtitle.innerText = item.querySelector('span').innerText + " Overview";
        });
    });

    // Theme Toggle Logic
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.remove('light-theme');
                document.body.classList.add('dark-theme');
            } else {
                document.body.classList.remove('dark-theme');
                document.body.classList.add('light-theme');
            }
        });
    }

    try {
        const response = await fetch('/api/data');
        const data = await response.json();

        // Populate KPIs
        // Populate KPIs (Old Dashboard grid)
        document.getElementById('kpi-sarimax-rmse').innerText = data.metrics.rmse.sarimax.toLocaleString();
        document.getElementById('kpi-xgboost-rmse').innerText = data.metrics.rmse.xgboost.toLocaleString();
        document.getElementById('kpi-lstm-rmse').innerText = data.metrics.rmse.lstm.toLocaleString();

        // Populate KPIs (New Image Components)
        document.getElementById('kpi-total').innerText = data.kpis.total_passengers;
        document.getElementById('kpi-avg').innerText = data.kpis.avg_monthly;
        document.getElementById('kpi-fuel').innerHTML = data.kpis.fuel_price + '<span class="kpi-unit">/gal</span>';
        document.getElementById('kpi-corr').innerText = 'R² = ' + data.kpis.correlation;

        /* --- NEW CHART FROM SCREENSHOT --- */
        const ctxHist = document.getElementById('historicalForecastChart').getContext('2d');
        
        let histLen = data.historical_len;
        let histData = data.historical_data.slice(); 
        while (histData.length < data.historical_dates.length) histData.push(null);
        let predData = Array(histLen - 1).fill(null);
        predData.push(data.historical_data[histLen - 1]);
        predData = predData.concat(data.predicted_data);

        const gradientPred = ctxHist.createLinearGradient(0, 0, 0, 400);
        gradientPred.addColorStop(0, 'rgba(34, 197, 94, 0.3)');
        gradientPred.addColorStop(1, 'rgba(34, 197, 94, 0.0)');

        new Chart(ctxHist, {
            type: 'line',
            data: {
                labels: data.historical_dates,
                datasets: [
                    { label: 'Historical Traffic (Millions)', data: histData, borderColor: '#3B82F6', borderWidth: 2, fill: false, pointRadius: 0, tension: 0.4 },
                    { label: 'Predicted Traffic (LSTM)', data: predData, borderColor: '#22C55E', borderWidth: 2, borderDash: [5, 5], backgroundColor: gradientPred, fill: true, pointRadius: 0, tension: 0.4 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'top', labels: { boxWidth: 40, padding: 20 } } },
                scales: { 
                    y: { min: 0, max: 6, ticks: { stepSize: 1 }, title: { display: true, text: 'Passengers (Millions)' } },
                    x: { ticks: { maxTicksLimit: 8 } }
                }
            }
        });

        /* --- DASHBOARD VIEW CHARTS --- */
        const ctxForecast = document.getElementById('forecastChart').getContext('2d');
        const gradientActual = ctxForecast.createLinearGradient(0, 0, 0, 400);
        gradientActual.addColorStop(0, 'rgba(59, 130, 246, 0.5)');
        gradientActual.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        new Chart(ctxForecast, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    { label: 'Actual Passengers', data: data.actual, borderColor: cb_primary, backgroundColor: gradientActual, fill: true, borderWidth: 3, pointBackgroundColor: cb_primary },
                    { label: 'SARIMAX', data: data.sarimax, borderColor: cb_sarimax, borderDash: [5, 5], borderWidth: 2, pointBackgroundColor: cb_sarimax },
                    { label: 'XGBoost', data: data.xgboost, borderColor: cb_xgboost, borderDash: [5, 5], borderWidth: 2, pointBackgroundColor: cb_xgboost },
                    { label: 'LSTM (Optimal)', data: data.lstm, borderColor: cb_lstm, borderWidth: 3, pointBackgroundColor: cb_lstm, pointBorderColor: '#fff', pointBorderWidth: 2, fill: false }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } } },
                scales: { y: { ticks: { callback: value => (value / 1000000).toFixed(1) + 'M' } } }
            }
        });

        // Error Chart
        const ctxError = document.getElementById('errorChart').getContext('2d');
        new Chart(ctxError, {
            type: 'bar',
            data: {
                labels: ['SARIMAX', 'XGBoost', 'LSTM'],
                datasets: [{
                    label: 'MAE',
                    data: [data.metrics.mae.sarimax, data.metrics.mae.xgboost, data.metrics.mae.lstm],
                    backgroundColor: ['rgba(245, 158, 11, 0.8)', 'rgba(16, 185, 129, 0.8)', 'rgba(139, 92, 246, 0.8)'],
                    borderWidth: 0, borderRadius: 6
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { ticks: { callback: value => (value / 1000).toFixed(0) + 'K' } } }
            }
        });

        /* --- DETAILED FORECAST VIEW CHART --- */
        const ctxDetailed = document.getElementById('detailedForecastChart').getContext('2d');
        new Chart(ctxDetailed, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    { label: 'LSTM Predictions', data: data.lstm, borderColor: cb_lstm, padding: 10, tension: 0.1 },
                    { label: 'XGBoost Predictions', data: data.xgboost, borderColor: cb_xgboost, padding: 10, tension: 0.1 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index' },
            }
        });

        /* --- FUEL VIEW CHART --- */
        const ctxFuel = document.getElementById('fuelChart').getContext('2d');
        new Chart(ctxFuel, {
            type: 'line',
            data: {
                labels: data.fuel.dates,
                datasets: [{ label: 'Jet Fuel Price (US Gulf)', data: data.fuel.prices, borderColor: '#EF4444', backgroundColor: 'rgba(239, 68, 68, 0.2)', fill: true, tension: 0.4 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { title: { display: true, text: 'USD per Gallon' } } }
            }
        });

        /* --- GDP VIEW CHART --- */
        const ctxGdp = document.getElementById('gdpChart').getContext('2d');
        new Chart(ctxGdp, {
            type: 'bar',
            data: {
                labels: data.gdp.dates,
                datasets: [{ label: 'Real GDP (Billions USD)', data: data.gdp.values, backgroundColor: cb_xgboost, borderRadius: 4 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: false, min: 28000 } }
            }
        });

    } catch (err) {
        console.error("Error fetching or parsing data:", err);
    }
});
