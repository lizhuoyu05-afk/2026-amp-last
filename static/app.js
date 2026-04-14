const sequenceEl = document.getElementById('sequence');
const seqLenEl = document.getElementById('seqLen');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const batchBtn = document.getElementById('batchBtn');
const downloadBtn = document.getElementById('downloadBtn');
const csvFileEl = document.getElementById('csvFile');
const statusText = document.getElementById('statusText');
const probabilityEl = document.getElementById('probability');
const verdictEl = document.getElementById('verdict');
const barEl = document.getElementById('bar');
const batchSummaryEl = document.getElementById('batchSummary');
const batchTableBodyEl = document.getElementById('batchTableBody');

let latestBatchResults = [];

function normalizedLength(value) {
  return value.replace(/\s+/g, '').length;
}

function updateLength() {
  seqLenEl.textContent = `${normalizedLength(sequenceEl.value)} aa`;
}

function renderBatchTable(results) {
  if (!results.length) {
    batchTableBodyEl.innerHTML = '<tr><td colspan="5" class="empty">无批量数据</td></tr>';
    return;
  }

  const preview = results.slice(0, 30);
  batchTableBodyEl.innerHTML = preview.map((row) => `
    <tr>
      <td>${row.row}</td>
      <td class="seq-cell">${row.sequence || '-'}</td>
      <td>${row.probability === null ? '-' : Number(row.probability).toFixed(3)}</td>
      <td>${row.verdict || '-'}</td>
      <td>${row.status}</td>
    </tr>
  `).join('');
}

function downloadBatchCsv() {
  if (!latestBatchResults.length) return;

  const esc = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  const header = 'row,sequence,probability,verdict,status,error';
  const body = latestBatchResults.map((row) => (
    [row.row, row.sequence, row.probability, row.verdict, row.status, row.error].map(esc).join(',')
  )).join('\n');

  const blob = new Blob([`${header}\n${body}`], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'cytotoxicity_batch_results.csv';
  link.click();
  URL.revokeObjectURL(url);
}

sequenceEl.addEventListener('input', updateLength);
downloadBtn.addEventListener('click', downloadBatchCsv);

clearBtn.addEventListener('click', () => {
  sequenceEl.value = '';
  updateLength();
  probabilityEl.textContent = '--';
  verdictEl.textContent = '判定：等待输入';
  barEl.style.width = '0%';
  statusText.textContent = '已清空，可重新输入序列。';
});

predictBtn.addEventListener('click', async () => {
  const sequence = sequenceEl.value.trim();
  if (!sequence) {
    statusText.textContent = '请先输入序列。';
    return;
  }

  statusText.textContent = '正在连接远程服务器并预测...';
  predictBtn.disabled = true;

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || '预测失败');
    }

    const prob = Number(data.probability || 0);
    probabilityEl.textContent = prob.toFixed(3);
    verdictEl.textContent = `判定：${data.verdict}`;
    barEl.style.width = `${Math.round(prob * 100)}%`;
    statusText.textContent = '单条预测完成。';
  } catch (err) {
    statusText.textContent = `错误：${err.message}`;
  } finally {
    predictBtn.disabled = false;
  }
});

batchBtn.addEventListener('click', async () => {
  if (!csvFileEl.files.length) {
    statusText.textContent = '请先选择 CSV 文件。';
    return;
  }

  const formData = new FormData();
  formData.append('file', csvFileEl.files[0]);

  statusText.textContent = '批量预测进行中，请稍候...';
  batchBtn.disabled = true;

  try {
    const response = await fetch('/api/predict-batch', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || '批量预测失败');
    }

    latestBatchResults = data.results || [];
    renderBatchTable(latestBatchResults);
    downloadBtn.disabled = latestBatchResults.length === 0;

    batchSummaryEl.textContent = `批量结果：总计 ${data.total}，成功 ${data.success}，失败 ${data.failed}（识别列：${data.sequence_column}）`;
    statusText.textContent = '批量预测完成。';
  } catch (err) {
    statusText.textContent = `批量错误：${err.message}`;
  } finally {
    batchBtn.disabled = false;
  }
});

updateLength();
