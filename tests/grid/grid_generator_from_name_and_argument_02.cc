// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2019 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

// Test GridGenerator::generate_from_name_and_arguments.

#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

template <int dim, int spacedim>
void
test(const std::string &name, const std::string &arguments)
{
  Triangulation<dim, spacedim> tria;

  deallog << "Generating Triangulation<" << dim << ", " << spacedim
          << "> : " << name << '(' << arguments << ')' << std::endl;

  GridGenerator::generate_from_name_and_arguments(tria, name, arguments);

  GridOut go;
  go.write_msh(tria, deallog.get_file_stream());

  std::ofstream ofile(name + "_" + std::to_string(dim) + "_" +
                      std::to_string(spacedim) + ".msh");
  go.write_msh(tria, ofile);
}

int
main()
{
  initlog();

  test<2, 3>("hyper_sphere", "0,0,0 : 1");

  test<2, 2>("quarter_hyper_ball", "0,0 : 1");
  test<3, 3>("quarter_hyper_ball", "0,0,0 : 1");

  test<2, 2>("hyper_ball_balanced", "0,0 : 1");
  test<3, 3>("hyper_ball_balanced", "0,0,0 : 1");

  test<2, 2>("subdivided_hyper_cube_with_simplices",
             "2 : 0.0 : 1.0 : false : false");
  test<3, 3>("subdivided_hyper_cube_with_simplices",
             "2 : 0.0 : 1.0 : false : false");

  test<2, 2>("subdivided_hyper_rectangle_with_simplices",
             "2, 2 : 0.0, 0.0 : 1.0, 2.0 : false : false");
  test<3, 3>("subdivided_hyper_rectangle_with_simplices",
             "2, 2, 3 : 0.0, 0.0, 1.0 : 1.0, 2.0, 3.0 : false : false");

  test<2, 2>("subdivided_hyper_L", "5, 5 : 0, 0 : 1, 1 : 2, 3");
  test<3, 3>("subdivided_hyper_L", "5, 5, 5 : 0, 0, 0 : 1, 1, 1 : 2, 2, 3");
}
