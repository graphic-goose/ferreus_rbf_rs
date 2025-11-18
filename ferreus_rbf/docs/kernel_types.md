Implemented kernel functions

# Linear radial basis function:

<div>
\[
\varphi(r) = -r
\]
</div>

# Thin plate spline radial basis function:

<div>
\[
\varphi(r)=
\begin{cases}
0, & r=0,\\
r^{2}\log r, & r>0.
\end{cases}
\]
</div>

# Cubic radial basis function:

<div>
\[
\varphi(r) = r^3
\]
</div>

# Spheroidal radial basis function:

<div>
\[
\varphi(r)=
s\begin{cases}
1-\lambda_{m}\,r_{s}, & r_{s}\le x^{*}_{m},\\
c_{m}^{-1}\,(1+r_{s}^{2})^{-m/2}, & r_{s}\ge x^{*}_{m}.
\end{cases}
\]
</div>

where   
<div>
\[
r_{s} = \kappa_{m}{r / R}
\]
</div>

with

- s = total sill
- R = base range

The Spheroidal family of covariance functions have the same
definition, with varying constant parameters based on the selected
order.

The order determines how steeply the interpolant asymptotically approaches `0.0`.
A higher order value gives more weighting to points at intermediate distances,
compared with lower orders.

The Spheroidal covariance function is a piecewise function that combines the linear
RBF function up to the inflexion point, and a scaled inverse multiquadric function
after that.

More information can be found [here](https://www.seequent.com/the-spheroidal-family-of-variograms-explained/).

## Parameters for each supported spheroidal order

<div style="width: 100%;">
  <table style="width: 100%; border-collapse: collapse;">
    <caption style="font-size: 1.2em;">
        Constant parameters for each supported spheroidal order
    </caption>
    <thead>
      <tr>
        <th style="text-align:left;">Order (<span>\(m\)</span>)</th>
        <th style="text-align:right;">3</th>
        <th style="text-align:right;">5</th>
        <th style="text-align:right;">7</th>
        <th style="text-align:right;">9</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Inflexion point (<span>\(x^{*}_{m}\)</span>)</td>
        <td style="text-align:right;">0.5000000000</td>
        <td style="text-align:right;">0.4082482905</td>
        <td style="text-align:right;">0.3535533906</td>
        <td style="text-align:right;">0.3162277660</td>
      </tr>
      <tr>
        <td>Y-intercept (<span>\(c_{m}\)</span>)</td>
        <td style="text-align:right;">1.1448668044</td>
        <td style="text-align:right;">1.1660474725</td>
        <td style="text-align:right;">1.1771820863</td>
        <td style="text-align:right;">1.1840505048</td>
      </tr>    
      <tr>
        <td>Linear slope (<span>\(\lambda_{m}\)</span>)</td>
        <td style="text-align:right;">0.7500000000</td>
        <td style="text-align:right;">1.0206207262</td>
        <td style="text-align:right;">1.2374368671</td>
        <td style="text-align:right;">1.4230249471</td>
      </tr>
      <tr>
        <td>Range scaling (<span>\(\kappa_{m}\)</span>)</td>
        <td style="text-align:right;">2.6798340586</td>
        <td style="text-align:right;">1.5822795750</td>
        <td style="text-align:right;">1.2008676644</td>
        <td style="text-align:right;">1.0000000000</td>
      </tr>
    </tbody>
  </table>
</div>
