/**
 * Visualization module using Plotly.js
 * Mimics the visualizations from the Streamlit app
 */

const Visualization = {
  /**
   * Create all visualizations for the analysis results
   */
  renderAnalysisVisualizations(results, metricsDef) {
    this.createGroupDistributionChart('viz-group-distribution', results.summary);
    this.createFairnessDashboard('viz-fairness-dashboard', results.metrics);
    this.createMetricComparisonChart('viz-metric-comparison', results.metrics, metricsDef);
    this.createDetailedMetricCards('viz-detailed-metrics', results.metrics, metricsDef);
  },

  /**
   * Render comparison between original and mitigated results
   */
  renderMitigationComparison(originalResults, mitigatedResults, metricsDef) {
    this.createComparisonChart('viz-mitigation-comparison', originalResults.metrics, mitigatedResults.metrics, metricsDef);
  },

  /**
   * Group Distribution Chart (Bar)
   */
  createGroupDistributionChart(elementId, summary) {
    const trace = {
      x: ['Privileged Group', 'Unprivileged Group'],
      y: [summary.privileged_count, summary.unprivileged_count],
      type: 'bar',
      marker: {
        color: ['#2ecc71', '#e74c3c']
      }
    };

    const layout = {
      title: 'Protected Attribute Group Distribution',
      xaxis: { title: 'Group' },
      yaxis: { title: 'Count' },
      margin: { t: 40, b: 40, l: 40, r: 40 },
      height: 350,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
  },

  /**
   * Fairness Dashboard (Pie Chart)
   */
  createFairnessDashboard(elementId, metricsResults) {
    let fairCount = 0;
    let biasedCount = 0;

    Object.values(metricsResults).forEach(m => {
      if (m.is_fair) fairCount++;
      else biasedCount++;
    });

    const data = [{
      values: [fairCount, biasedCount],
      labels: ['Fair Metrics', 'Biased Metrics'],
      type: 'pie',
      hole: .4,
      marker: {
        colors: ['#2ecc71', '#e74c3c']
      },
      textinfo: 'label+percent'
    }];

    const layout = {
      title: 'Overall Fairness Assessment',
      height: 350,
      margin: { t: 40, b: 20, l: 20, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      annotations: [
        {
          font: { size: 20 },
          showarrow: false,
          text: `${fairCount}/${fairCount + biasedCount}`,
          x: 0.5,
          y: 0.5
        }
      ]
    };

    Plotly.newPlot(elementId, data, layout, { responsive: true, displayModeBar: false });
  },

  /**
   * Metric Comparison Chart (Horizontal Bar)
   */
  createMetricComparisonChart(elementId, metricsResults, metricsDef) {
    const names = [];
    const values = [];
    const colors = [];
    const hoverTexts = [];

    Object.entries(metricsResults).forEach(([key, result]) => {
      const def = metricsDef[key] || { name: key };
      names.push(def.name);

      const val = result.value;
      // Handle N/A or non-numeric
      if (typeof val === 'number') {
        values.push(Math.abs(val));
        colors.push(result.is_fair ? '#2ecc71' : '#e74c3c');
        hoverTexts.push(`Value: ${val.toFixed(4)}<br>${result.interpretation || ''}`);
      } else {
        values.push(0);
        colors.push('#95a5a6');
        hoverTexts.push('N/A');
      }
    });

    const trace = {
      type: 'bar',
      x: values,
      y: names,
      orientation: 'h',
      marker: { color: colors },
      text: values.map(v => v ? v.toFixed(3) : 'N/A'),
      textposition: 'auto',
      hoverinfo: 'text',
      hovertext: hoverTexts
    };

    const layout = {
      title: 'Metric Values (Absolute)',
      xaxis: { title: 'Absolute Value' },
      yaxis: { automargin: true },
      margin: { t: 40, b: 40, l: 150, r: 20 },
      height: Math.max(400, names.length * 50),
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
  },

  /**
   * Create detailed cards with Gauge charts for each metric
   */
  createDetailedMetricCards(containerId, metricsResults, metricsDef) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    Object.entries(metricsResults).forEach(([key, result]) => {
      const def = metricsDef[key] || { name: key };

      const card = document.createElement('div');
      card.className = 'metric-card-detail';

      // Status header
      const statusClass = result.is_fair ? 'status-fair' : 'status-biased';
      const statusIcon = result.is_fair ? '✅' : '⚠️';
      const statusText = result.is_fair ? 'Fair' : 'Biased';

      card.innerHTML = `
        <div class="metric-header">
          <h4>${def.name}</h4>
          <span class="badge ${statusClass}">${statusIcon} ${statusText}</span>
        </div>
        <div class="metric-body">
          <div class="metric-info">
            <p><strong>Value:</strong> ${typeof result.value === 'number' ? result.value.toFixed(4) : 'N/A'}</p>
            <p><small>${def.description || ''}</small></p>
            <div class="interpretation">${result.interpretation || ''}</div>
          </div>
          <div class="metric-chart" id="gauge-${key}"></div>
        </div>
      `;

      container.appendChild(card);

      // Create Gauge Chart
      if (typeof result.value === 'number') {
        this.createGaugeChart(`gauge-${key}`, result.value, key);
      }
    });
  },

  createGaugeChart(elementId, value, key) {
    // Determine range based on metric type (similar to Streamlit logic)
    const isRatio = ['disparate_impact', 'positive_rate_ratio'].includes(key);

    let gauge = {};

    if (isRatio) {
      // Ideal around 1.0
      gauge = {
        axis: { range: [0, 2] },
        bar: { color: "#2c3e50" },
        steps: [
          { range: [0, 0.8], color: "rgba(231, 76, 60, 0.3)" }, // red
          { range: [0.8, 1.25], color: "rgba(46, 204, 113, 0.3)" }, // green
          { range: [1.25, 2], color: "rgba(231, 76, 60, 0.3)" } // red
        ],
        threshold: {
          line: { color: "red", width: 4 },
          thickness: 0.75,
          value: 1.0
        }
      };
    } else {
      // Ideal around 0
      gauge = {
        axis: { range: [-1, 1] },
        bar: { color: "#2c3e50" },
        steps: [
          { range: [-1, -0.1], color: "rgba(231, 76, 60, 0.3)" },
          { range: [-0.1, 0.1], color: "rgba(46, 204, 113, 0.3)" },
          { range: [0.1, 1], color: "rgba(231, 76, 60, 0.3)" }
        ],
        threshold: {
          line: { color: "red", width: 4 },
          thickness: 0.75,
          value: 0
        }
      };
    }

    const data = [{
      type: "indicator",
      mode: "gauge+number",
      value: value,
      gauge: gauge
    }];

    const layout = {
      margin: { t: 20, b: 20, l: 20, r: 20 },
      height: 150,
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { size: 10 }
    };

    Plotly.newPlot(elementId, data, layout, { responsive: true, displayModeBar: false });
  },

  /**
   * Comparison Chart: Original vs Mitigated
   * Uses subplots to handle different scales and shows acceptable ranges
   */
  createComparisonChart(elementId, originalMetrics, mitigatedMetrics, metricsDef) {
    const keys = Array.from(new Set([...Object.keys(originalMetrics), ...Object.keys(mitigatedMetrics)]));
    const numSubplots = keys.length;

    if (numSubplots === 0) return;

    const traces = [];
    const layout = {
      title: 'Mitigation Analysis: Before vs After (with Acceptable Ranges)',
      height: Math.max(400, numSubplots * 200),
      grid: { rows: numSubplots, columns: 1, pattern: 'independent' },
      showlegend: true,
      legend: { orientation: 'h', y: 1.05 },
      margin: { t: 80, b: 50, l: 60, r: 20 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      shapes: []
    };

    keys.forEach((key, index) => {
      const def = metricsDef[key] || { name: key };
      const subplotIdx = index + 1;

      const v1 = originalMetrics[key]?.value;
      const v2 = mitigatedMetrics[key]?.value;

      // Original trace
      traces.push({
        x: ['Original', 'Mitigated'],
        y: [typeof v1 === 'number' ? v1 : null, typeof v2 === 'number' ? v2 : null],
        name: `${def.name} (Original)`,
        type: 'bar',
        marker: { color: index % 2 === 0 ? '#e74c3c' : '#c0392b' },
        xaxis: `x${subplotIdx}`,
        yaxis: `y${subplotIdx}`,
        width: 0.5,
        hoverinfo: 'y+name',
        showlegend: index === 0 // Show only one "Original" legend entry if colors were consistent, but here they differ per subplot
      });

      // Update trace colors to be consistent for categories
      traces[traces.length - 1].name = 'Original';
      traces[traces.length - 1].marker.color = '#e74c3c';
      traces[traces.length - 1].showlegend = (index === 0);

      // Add mitigated as a second bar in the same subplot
      // Actually, let's use two marks per subplot
      const trace2 = {
        x: ['Original', 'Mitigated'],
        y: [null, typeof v2 === 'number' ? v2 : null],
        name: 'Mitigated',
        type: 'bar',
        marker: { color: '#2ecc71' },
        xaxis: `x${subplotIdx}`,
        yaxis: `y${subplotIdx}`,
        width: 0.5,
        hoverinfo: 'y+name',
        showlegend: (index === 0)
      };

      // Update original trace to only show one bar
      traces[traces.length - 1].y = [typeof v1 === 'number' ? v1 : null, null];
      traces.push(trace2);

      // Add Fair Range Shading
      const threshold = def.threshold || {};
      if (typeof threshold.min === 'number' && typeof threshold.max === 'number') {
        layout.shapes.push({
          type: 'rect',
          xref: `x${subplotIdx}`,
          yref: `y${subplotIdx}`,
          x0: -0.5,
          x1: 1.5,
          y0: threshold.min,
          y1: threshold.max,
          fillcolor: 'rgba(46, 204, 113, 0.15)',
          line: { width: 0 },
          layer: 'below'
        });

        // Add target line
        let ideal = 0;
        if (def.ideal_value && (def.ideal_value.includes('1.0') || def.ideal_value.includes('1.00'))) {
          ideal = 1.0;
        }

        layout.shapes.push({
          type: 'line',
          xref: `x${subplotIdx}`,
          yref: `y${subplotIdx}`,
          x0: -0.5,
          x1: 1.5,
          y0: ideal,
          y1: ideal,
          line: { color: 'rgba(46, 204, 113, 0.5)', width: 2, dash: 'dash' }
        });
      }

      // Subplot Axis Titles
      layout[`xaxis${subplotIdx}`] = {
        showticklabels: true,
        fixedrange: true
      };
      layout[`yaxis${subplotIdx}`] = {
        title: { text: def.name, font: { size: 10 } },
        fixedrange: true
      };

      // Adjust Y axis range to ensure the shape is visible
      if (typeof threshold.min === 'number') {
        const minVal = Math.min(v1 || 0, v2 || 0, threshold.min);
        const maxVal = Math.max(v1 || 0, v2 || 0, threshold.max);
        const padding = (maxVal - minVal) * 0.2 || 0.1;
        layout[`yaxis${subplotIdx}`].range = [minVal - padding, maxVal + padding];
      }
    });

    Plotly.newPlot(elementId, traces, layout, { responsive: true, displayModeBar: false });
  }
};
