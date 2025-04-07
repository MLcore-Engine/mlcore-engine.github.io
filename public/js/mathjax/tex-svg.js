/*************************************************************************
 *
 *  MathJax v3.2.0 - Basic file to load full MathJax and
 *                  convert TeX notation to SVG output
 *
 *  Copyright (c) 2021 The MathJax Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// 简单的实现，加载真实的MathJax库
(function () {
  // 创建加载脚本元素
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.async = true;
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
  
  // 在加载失败时的备用CDN
  script.onerror = function() {
    var backupScript = document.createElement('script');
    backupScript.type = 'text/javascript';
    backupScript.async = true;
    backupScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-svg.min.js';
    document.head.appendChild(backupScript);
  };
  
  // 添加到文档
  document.head.appendChild(script);
})(); 