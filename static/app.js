const sequenceEl = document.getElementById('sequence');
const seqLenEl = document.getElementById('seqLen');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const statusText = document.getElementById('statusText');
const probabilityEl = document.getElementById('probability');
const verdictEl = document.getElementById('verdict');
const barEl = document.getElementById('bar');

function normalizedLength(value) {
  return value.replace(/\s+/g, '').length;
}

function updateLength() {
  seqLenEl.textContent = `${normalizedLength(sequenceEl.value)} aa`;
}

sequenceEl.addEventListener('input', updateLength);

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
    statusText.textContent = '预测完成。';
  } catch (err) {
    statusText.textContent = `错误：${err.message}`;
  } finally {
    predictBtn.disabled = false;
  }
});

updateLength();
