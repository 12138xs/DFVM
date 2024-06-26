<h1 align="center">Deep Finite Volume Method</h1>

<div align="center">
    <a href="https://arxiv.org/abs/2305.06863">Paper (arXiv)</a> | 
    <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4735856">Paper (SSRN)</a>
</div>

<div align="center">
    Jianhuan Cen, and Qingsong Zou</br>
    Sun Yat-sen University
</div>

<div></br></div>

<!-- ![flow-examples](figs/flow-examples.png) -->

This is the code for the paper: [Deep Finite Volume Method for Partial Differential Equations](https://arxiv.org/abs/2305.06863).

DFVM centers on a novel loss function crafted from local conservation laws derived from the original PDE, distinguishing DFVM from traditional deep learning methods. By formulating DFVM in the weak form of the PDE rather than the strong form, we enhance accuracy, particularly beneficial for PDEs with less smooth solutions compared to strong-form-based methods like Physics-Informed Neural Networks (PINNs). A key technique of DFVM lies in its transformation of all second-order or higher derivatives of neural networks into first-order derivatives which can be comupted directly using Automatic Differentiation (AD). This adaptation significantly reduces computational overhead, particularly advantageous for solving high-dimensional PDEs.


## Citation

```
@article{cen2024dfvm,
      title={Deep Finite Volume Method for High-Dimensional Partial Differential Equations}, 
      author={Jianhuan Cen and Qingsong Zou},
      url={https://arxiv.org/abs/2305.06863}, 
      year={2024},
}
```