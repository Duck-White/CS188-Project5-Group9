/* CS188 Project 5 - real inference in browser from exported PyTorch tensors.
 * No artificial labels; if no real checkpoint exists, Q3/Q4/Q5 remain unavailable.
 */
(function (scope) {
  'use strict';
  function params(bundle) {
    if (!bundle || bundle.format !== 'cs188-p5-tensor-v1' || !bundle.tensors) throw Error('Sai định dạng trọng số.');
    return bundle.tensors;
  }
  function linear(input, tensors, prefix) {
    const W=tensors[prefix+'.weight'], B=tensors[prefix+'.bias'];
    if(!W || W.shape.length!==2) throw Error('Sai cấu trúc layer '+prefix);
    const O=W.shape[0], I=W.shape[1];
    if(input.length!==I) throw Error('Sai số chiều '+prefix+': '+input.length+' != '+I);
    const output=new Float64Array(O);
    for(let o=0;o<O;o++){
      let sum=B ? B.data[o] : 0, start=o*I;
      for(let i=0;i<I;i++)sum+=W.data[start+i]*input[i];
      output[o]=sum;
    }
    return output;
  }
  function relu(values) {return Float64Array.from(values, x=>Math.max(0,x));}
  function softmax(logits) {
    const maximum=Math.max(...logits),ex=Array.from(logits,x=>Math.exp(x-maximum));
    const total=ex.reduce((a,b)=>a+b,0);return ex.map(x=>x/total);
  }
  function digitQ3(pixels,bundle){
    if(pixels.length!==784)throw Error('Q3 cần đúng 784 pixel.');
    const t=params(bundle);let x=relu(linear(pixels,t,'fc1'));x=relu(linear(x,t,'fc2'));return softmax(linear(x,t,'fc3'));
  }
  function regressionQ2(x,bundle){
    const t=params(bundle);
    let v=relu(linear([x],t,'fc1'));v=relu(linear(v,t,'fc2'));
    return linear(v,t,'fc3')[0];
  }
  function languageQ4(word,bundle){
    const t=params(bundle),alpha=bundle.alphabet;
    if(!Array.isArray(alpha) || alpha.length!==47)throw Error('Thiếu bảng mã hóa Q4 chính thức.');
    const idx=new Map(alpha.map((x,i)=>[x,i]));let h=new Float64Array(256);
    const letters=Array.from(word.trim().toLowerCase());if(!letters.length)throw Error('Nhập một từ.');
    const input=t['input_layer.weight'], inputBias=t['input_layer.bias'];
    for(const ch of letters){
      if(!idx.has(ch))throw Error('Ký tự không thuộc bảng 47 ký tự: '+ch);
      const j=idx.get(ch), hh=linear(h,t,'hidden_layer');
      for(let k=0;k<h.length;k++)hh[k]=Math.max(0,hh[k]+input.data[k*47+j]+inputBias.data[k]);
      h=hh;
    }
    const x=relu(linear(h,t,'fc1'));return softmax(linear(x,t,'fc2'));
  }
  function digitQ5(pixels,bundle){
    if(pixels.length!==784)throw Error('Q5 cần đúng 784 pixel.');
    const t=params(bundle),kernel=t['convolution_weights'].data;
    const W1=t.w1, b1=t.b1.data, W2=t.w2, b2=t.b2.data;
    const O=200, map=new Float64Array(26*26);
    for(let y=0;y<26;y++)for(let x=0;x<26;x++){
      let sum=0;
      for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++)sum+=pixels[(y+ky)*28+x+kx]*kernel[ky*3+kx];
      map[y*26+x]=sum;
    }
    const hidden=new Float64Array(O);
    for(let o=0;o<O;o++){let sum=b1[o];for(let i=0;i<676;i++)sum+=map[i]*W1.data[i*O+o];hidden[o]=Math.max(0,sum)}
    const logits=new Float64Array(10);
    for(let o=0;o<10;o++){let sum=b2[o];for(let i=0;i<O;i++)sum+=hidden[i]*W2.data[i*10+o];logits[o]=sum}
    return softmax(logits);
  }
  function attentionQ6(tokens,bundle){
    const t=params(bundle);const queries=tokens.map(x=>linear(x,t,'q_layer'));
    const keys=tokens.map(x=>linear(x,t,'k_layer'));
    const values=tokens.map(x=>linear(x,t,'v_layer'));
    const T=tokens.length,D=tokens[0].length,weights=[],output=[];
    for(let i=0;i<T;i++){
      const score=[];for(let j=0;j<=i;j++){
        let p=0;for(let k=0;k<D;k++)p+=queries[i][k]*keys[j][k];score.push(p/Math.sqrt(D));
      }
      const row=softmax(score).concat(Array(T-i-1).fill(0));weights.push(row);
      const result=[];for(let k=0;k<D;k++){let v=0;for(let j=0;j<=i;j++)v+=row[j]*values[j][k];result.push(v)}output.push(result);
    }
    return {attention:weights,output};
  }
  scope.CS188Browser={digitQ3,regressionQ2,languageQ4,digitQ5,attentionQ6,softmax};
})(globalThis);