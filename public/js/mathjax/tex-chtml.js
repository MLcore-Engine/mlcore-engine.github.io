/**
 * MathJax v3 - Local version
 * Handling LaTeX math in HTML
 */
(function () {
  // 配置对象
  const config = {
    jax: ["input/TeX", "output/CommonHTML"],
    extensions: ["tex2jax.js", "MathMenu.js", "MathZoom.js"],
    tex2jax: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    TeX: {
      extensions: ["AMSmath.js", "AMSsymbols.js", "noErrors.js", "noUndefined.js"]
    }
  };

  // 查找页面中的所有数学公式
  function findMath() {
    const mathElements = [];
    const inlineMathPattern1 = /\$([^\$]+)\$/g;
    const inlineMathPattern2 = /\\[\(]([^\\]+)\\[\)]/g;
    const displayMathPattern1 = /\$\$([^\$]+)\$\$/g;
    const displayMathPattern2 = /\\[\[]([^\\]+)\\[\]]/g;

    // 处理页面中的所有文本节点
    function processTextNode(node) {
      const text = node.textContent;
      let match;
      
      // 查找内联公式
      inlineMathPattern1.lastIndex = 0;
      while ((match = inlineMathPattern1.exec(text)) !== null) {
        replaceMath(node, match[0], match[1], false);
      }
      
      inlineMathPattern2.lastIndex = 0;
      while ((match = inlineMathPattern2.exec(text)) !== null) {
        replaceMath(node, match[0], match[1], false);
      }
      
      // 查找块级公式
      displayMathPattern1.lastIndex = 0;
      while ((match = displayMathPattern1.exec(text)) !== null) {
        replaceMath(node, match[0], match[1], true);
      }
      
      displayMathPattern2.lastIndex = 0;
      while ((match = displayMathPattern2.exec(text)) !== null) {
        replaceMath(node, match[0], match[1], true);
      }
    }

    // 替换找到的公式
    function replaceMath(node, fullMatch, formula, isDisplay) {
      const span = document.createElement('span');
      span.className = isDisplay ? 'math-display' : 'math-inline';
      span.style.fontFamily = 'math, serif';
      span.style.fontStyle = 'italic';
      
      if (isDisplay) {
        span.style.display = 'block';
        span.style.textAlign = 'center';
        span.style.margin = '1em 0';
      } else {
        span.style.display = 'inline';
      }
      
      span.textContent = formula;
      span.setAttribute('data-math-formula', formula);
      
      // 将公式节点添加到数组
      mathElements.push({
        element: span,
        formula: formula,
        isDisplay: isDisplay
      });
      
      node.parentNode.replaceChild(span, node);
    }

    // 遍历文档中的所有文本节点
    function walkDOM(node) {
      if (node.nodeType === 3) { // 文本节点
        processTextNode(node);
      } else if (node.nodeType === 1) { // 元素节点
        // 跳过脚本、样式等特殊标签
        const tagName = node.nodeName.toLowerCase();
        if (['script', 'noscript', 'style', 'textarea', 'pre', 'code'].indexOf(tagName) === -1) {
          // 处理子节点
          for (let i = 0; i < node.childNodes.length; i++) {
            walkDOM(node.childNodes[i]);
          }
        }
      }
    }

    walkDOM(document.body);
    return mathElements;
  }

  // 初始化函数
  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', findMath);
    } else {
      findMath();
    }
  }

  // 启动
  init();
})(); 