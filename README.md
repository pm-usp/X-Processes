# X-Processes

**EDOC 2021 version (older version)**: run "python main.py" (edit/change parameters inside main.py file).
- Associate paper: Marcelo Fantinato, Sarajane Marques Peres, Hajo A. Reijers, X-Processes: Discovering More Accurate Business Process Models with a Genetic Algorithms Method, 25th IEEE International Enterprise Distributed Object Computing Conference, pp. 114-123, 2021. https://doi.org/10.1109/EDOC52215.2021.00022

**IS 2023 version (newer version)**: run "python xprocesses.py -h" for help using the method.
- Associate article: Marcelo Fantinato, Sarajane Marques Peres, Hajo A. Reijers, X-Processes: Process model discovery with the best balance among fitness, precision, simplicity, and generalization through a genetic algorithm, Information Systems 119, 2023. https://doi.org/10.1016/j.is.2023.102247

**Sampling (newest version)**: run "python xprocesses.py -h" for help using the method.
- This version introduces several new and updated features, including the option to use sampling for faster execution. If you use this method/version, please cite our IS 2023 article.
  
# Requirements: 
- The requirements.txt refers only to the "Sampling" newest version.

# Licensing and third-party components

This project is released under the terms of the GNU General Public License v3.0 or (at your option) any later version. See the LICENSE.GPL-3 file for details.

This repository also includes a Java archive (`java/jbpt_mini-1.0.jar`) that is based on code distributed under the GNU Lesser General Public License v3.0 (LGPLv3). This `.jar` is executed via a runtime subprocess and may be replaced by a modified version that is interface-compatible, as allowed under section 4 of the LGPLv3. See https://github.com/jbpt for more details. As required by the LGPLv3, the `jbpt_mini-1.0.jar` component may be replaced with a modified version, provided that it remains interface-compatible. The software is designed to support such substitution.

This project also includes several Java `.jar` files located under `java/lib/`, which are derived from the [bpmtk project](https://github.com/nemo-91/bpmtk) and originally from the [ProM framework](https://www.promtools.org/). These libraries are redistributed under the terms of the GNU General Public License (GPL). See the LICENSE.GPL-3 file for details.

You should have received copies of both the GNU General Public License and the GNU Lesser General Public License along with this project. If not, see:
- https://www.gnu.org/licenses/gpl-3.0.html
- https://www.gnu.org/licenses/lgpl-3.0.html
