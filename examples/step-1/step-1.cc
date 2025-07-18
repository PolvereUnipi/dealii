// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2022 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

// This test provides a minimal Laplace equation in 2D for testing
// the BDDC preconditioner. It cannot be simpler as it fails to setup
// in simpler matrices (i.e. diagonal), and the preset matrix factories
// create non-IS matrices, which are not supported by BDDC.

#include <deal.II/base/index_set.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>

#include "../../tests/tests.h"

// Vectors:
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

// Block Matrix and Vectors:
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>

// Dof and sparsity tools:
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>



// int main(int argc, char *argv[])
// {
//   using size_type = PETScWrappers::MPI::SparseMatrix::size_type;

//   Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

//   initlog();
//   std::cout << std::setprecision(10);
//   parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
//   FE_Q<2>                                 fe(1);
//   DoFHandler<2>                           dof_handler(triangulation);
//   GridGenerator::hyper_cube(triangulation);
//   triangulation.refine_global(3);
//   // triangulation.begin_active()->set_refine_flag();
//   // triangulation.execute_coarsening_and_refinement();
//   // triangulation.refine_global(1);

//   dof_handler.distribute_dofs(fe);

//   const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
//   const IndexSet  locally_relevant_dofs =
//     DoFTools::extract_locally_relevant_dofs(dof_handler);
//   const IndexSet locally_active_dofs =
//     DoFTools::extract_locally_active_dofs(dof_handler);

//   DynamicSparsityPattern dsp(locally_relevant_dofs);

//   DoFTools::make_sparsity_pattern(dof_handler, dsp);
//   SparsityTools::distribute_sparsity_pattern(dsp,
//                                              dof_handler.locally_owned_dofs(),
//                                              MPI_COMM_WORLD,
//                                              locally_relevant_dofs);

//   PETScWrappers::MPI::SparseMatrix system_matrix;
//   system_matrix.reinit(locally_owned_dofs,
//                        locally_active_dofs,
//                        locally_owned_dofs,
//                        locally_active_dofs,
//                        dsp,
//                        MPI_COMM_WORLD);



//   std::cout << "MATIS:OK" << std::endl;
//   PETScWrappers::MPI::Vector locally_relevant_solution(locally_owned_dofs,
//                                                        locally_relevant_dofs,
//                                                        MPI_COMM_WORLD);
//   PETScWrappers::MPI::Vector
//   completely_distributed_solution(locally_owned_dofs,
//                                                              MPI_COMM_WORLD);
//   PETScWrappers::MPI::Vector system_rhs(locally_owned_dofs, MPI_COMM_WORLD);

//   const QGauss<2> quadrature_formula(2);
//   FEValues<2>     fe_values(fe,
//                         quadrature_formula,
//                         update_values | update_gradients |
//                           update_quadrature_points | update_JxW_values);

//   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
//   const unsigned int n_q_points    = quadrature_formula.size();

//   FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
//   Vector<double>     cell_rhs(dofs_per_cell);

//   AffineConstraints<double> constraints;
//   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

//   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

//   for (const auto &cell : dof_handler.active_cell_iterators())
//     if (cell->is_locally_owned())
//       {
//         cell_matrix = 0.;
//         cell_rhs    = 0.;

//         fe_values.reinit(cell);

//         for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
//           {
//             const double rhs_value = 2.0;

//             for (unsigned int i = 0; i < dofs_per_cell; ++i)
//               {
//                 for (unsigned int j = 0; j < dofs_per_cell; ++j)
//                   {
//                     cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
//                                            fe_values.shape_grad(j, q_point) *
//                                            fe_values.JxW(q_point) +
//                                          fe_values.shape_value(i, q_point) *
//                                            fe_values.shape_value(j, q_point)
//                                            * fe_values.JxW(q_point);
//                   }

//                 cell_rhs(i) += 1.0 * fe_values.shape_value(i, q_point) *
//                                fe_values.JxW(q_point);
//               }
//           }

//         cell->get_dof_indices(local_dof_indices);
//         system_matrix.add(local_dof_indices, cell_matrix);
//         system_rhs.add(local_dof_indices, cell_rhs);
//         // constraints.distribute_local_to_global(
//         //   cell_matrix, cell_rhs, local_dof_indices, system_matrix,
//         //   system_rhs);
//       }


//   system_matrix.compress(VectorOperation::add);
//   system_rhs.compress(VectorOperation::add);

//   // Create a vector of ones
//   PETScWrappers::MPI::Vector vector_of_ones(locally_owned_dofs,
//   MPI_COMM_WORLD); vector_of_ones = 1.0;

//   // Multiply the system matrix by the vector of ones
//   PETScWrappers::MPI::Vector result_vector(locally_owned_dofs,
//   MPI_COMM_WORLD); system_matrix.vmult(result_vector, vector_of_ones);

//   // Compute and output the norm of the resulting vector
//   const double norm = result_vector.l2_norm();
//   std::cout << "Norm of the resulting vector: " << norm << std::endl;


//   // system_matrix.print(std::cout);
//   // std::cout << "Matrix norm = " << system_matrix.frobenius_norm() <<
//   // std::endl;

//   SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

//   PETScWrappers::SolverCG                            solver(solver_control);
//   PETScWrappers::PreconditionBDDC<2>                 preconditioner;
//   PETScWrappers::PreconditionBDDC<2>::AdditionalData data;

//   // Now we setup the dof coordinates if a sufficiently new PETSc is used
//   std::map<types::global_dof_index, Point<2>> dof_2_point;
//   DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
//                                        dof_handler,
//                                        dof_2_point);
//   std::vector<Point<2>> coords(locally_owned_dofs.n_elements());
//   unsigned int          k = 0;
//   for (auto it = locally_owned_dofs.begin(); it != locally_owned_dofs.end();
//        ++it, ++k)
//     {
//       coords[k] = dof_2_point[*it];
//     }
//   data.coords       = coords;
//   data.use_vertices = true;

//   preconditioner.initialize(system_matrix, data);
//   // check_solver_within_range(solver.solve(system_matrix,
//   //                                        completely_distributed_solution,
//   //                                        system_rhs,
//   //                                        preconditioner),
//   //                           solver_control.last_step(),
//   //                           1,
//   //                           2);
//   solver.solve(system_matrix,
//                completely_distributed_solution,
//                system_rhs,
//                preconditioner);
//   std::cout << "   Solved in " << solver_control.last_step()
//             << " iterations with residual of " << solver_control.last_value()
//             << std::endl;

//   // Copy solution into the locally relevant vector to output with ghost
//   cells locally_relevant_solution = completely_distributed_solution;

//   // Output solution to VTU file
//   DataOut<2> data_out;
//   data_out.attach_dof_handler(dof_handler);
//   data_out.add_data_vector(completely_distributed_solution, "solution");
//   data_out.build_patches();

//   const std::string filename =
//     "solution-" +
//     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
//     ".vtu";
//   std::ofstream output(filename);
//   data_out.write_vtu(output);

//   // Create master PVTU file that links all processor files
//   if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
//     {
//       std::vector<std::string> filenames;
//       for (unsigned int i = 0;
//            i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
//            ++i)
//         filenames.push_back("solution-" + Utilities::int_to_string(i, 4) +
//                             ".vtu");

//       const std::string master_filename = "solution.pvtu";
//       std::ofstream     master_output(master_filename);
//       data_out.write_pvtu_record(master_output, filenames);
//     }

//   std::cout << "CG/BDDC:OK" << std::endl;
//   return 0;
// }

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../tests/tests.h"


// Test that a coupling matrix can be constructed for each pair of dimension
// and immersed dimension, and check that quadratic functions are correctly
// projected.

template <int dim, int spacedim>
void test()
{
  const auto        &comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(comm) == 0));

  pcout << "dim: " << dim << ", spacedim: " << spacedim << std::endl;

  parallel::shared::Triangulation<dim, spacedim>           tria(comm);
  parallel::distributed::Triangulation<spacedim, spacedim> space_tria(comm);

  GridGenerator::hyper_cube(tria, -.4, .3);
  GridGenerator::hyper_cube(space_tria, -1, 1);

  tria.refine_global(5);
  space_tria.refine_global(5);

  FE_Q<dim, spacedim>      fe(2);
  FE_Q<spacedim, spacedim> space_fe(2);

  pcout << "FE      : " << fe.get_name() << std::endl
        << "Space FE: " << space_fe.get_name() << std::endl;

  DoFHandler<dim, spacedim>      dh(tria);
  DoFHandler<spacedim, spacedim> space_dh(space_tria);

  dh.distribute_dofs(fe);
  space_dh.distribute_dofs(space_fe);

  auto space_locally_owned_dofs = space_dh.locally_owned_dofs();
  auto locally_owned_dofs       = dh.locally_owned_dofs();


  pcout << "Dofs      : " << dh.n_dofs() << std::endl
        << "Space dofs: " << space_dh.n_dofs() << std::endl;

  std::cout << "Local dofs      : " << locally_owned_dofs.n_elements()
            << std::endl;
  std::cout << "Local space dofs: " << space_locally_owned_dofs.n_elements()
            << std::endl;
  QGauss<dim> quad(3); // Quadrature for coupling


  // TrilinosWrappers::SparsityPattern sparsity(space_locally_owned_dofs,
  //                                            locally_owned_dofs,
  //                                            comm);
  auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(space_dh);

  DynamicSparsityPattern sparsity(space_dh.n_dofs(),
                                  dh.n_dofs(),
                                  locally_relevant_dofs);

  NonMatching::create_coupling_sparsity_pattern(space_dh, dh, quad, sparsity);
  sparsity.compress();

  SparsityTools::distribute_sparsity_pattern(sparsity,
                                             space_locally_owned_dofs,
                                             comm,
                                             locally_relevant_dofs);

  auto space_locally_active_dofs =
    DoFTools::extract_locally_active_dofs(space_dh);
  auto locally_active_dofs_embedded = DoFTools::extract_locally_active_dofs(dh);

  PETScWrappers::MPI::SparseMatrix coupling;
  pcout << "Before coupling matrix reinit" << std::endl;
  // coupling.reinit(space_locally_owned_dofs, locally_owned_dofs, sparsity,
  // comm);
  coupling.reinit(space_locally_owned_dofs,
                  space_locally_active_dofs,
                  locally_owned_dofs,
                  locally_active_dofs_embedded,
                  sparsity,
                  comm);

  pcout << "Before coupling matrix creation" << std::endl;
  NonMatching::create_coupling_mass_matrix(space_dh, dh, quad, coupling);
  coupling.compress(VectorOperation::add);
  pcout << "Coupling matrix size: " << coupling.m() << " x " << coupling.n()
        << std::endl;

  auto embedded_locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dh);
  DynamicSparsityPattern mass_sparsity(embedded_locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dh, mass_sparsity);

  SparsityTools::distribute_sparsity_pattern(mass_sparsity,
                                             locally_owned_dofs,
                                             comm,
                                             embedded_locally_relevant_dofs);
  mass_sparsity.compress();
  PETScWrappers::MPI::SparseMatrix mass_matrix;
  mass_matrix.reinit(locally_owned_dofs, mass_sparsity, comm);

  {
    QGauss<dim>             quad(4);
    FEValues<dim, spacedim> fev(fe, quad, update_values | update_JxW_values);
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    FullMatrix<double>        cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    AffineConstraints<double> constraints;

    for (auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          fev.reinit(cell);
          cell->get_dof_indices(dofs);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
              for (unsigned int q = 0; q < quad.size(); ++q)
                cell_matrix(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
          constraints.distribute_local_to_global(cell_matrix,
                                                 dofs,
                                                 mass_matrix);
        }
    mass_matrix.compress(VectorOperation::add);
  }

  // now take the square function in space, project them onto the immersed
  // space, get back ones, and check for the error.
  PETScWrappers::MPI::Vector space_square(space_locally_owned_dofs, comm);
  PETScWrappers::MPI::Vector squares(locally_owned_dofs, comm);
  PETScWrappers::MPI::Vector Mprojected_squares(locally_owned_dofs, comm);
  PETScWrappers::MPI::Vector projected_squares(locally_owned_dofs, comm);

  VectorTools::interpolate(space_dh,
                           Functions::SquareFunction<spacedim>(),
                           space_square);
  VectorTools::interpolate(dh, Functions::SquareFunction<spacedim>(), squares);

  coupling.Tvmult(Mprojected_squares, space_square);

  SolverControl                        cn(100, 1e-12, false, false);
  PETScWrappers::SolverCG              solver(cn);
  PETScWrappers::PreconditionBoomerAMG prec;
  prec.initialize(mass_matrix);

  solver.solve(mass_matrix, projected_squares, Mprojected_squares, prec);

  pcout << "Squares norm    : " << projected_squares.l2_norm() << std::endl;

  projected_squares -= squares;

  pcout << "Error on squares: " << projected_squares.l2_norm() << std::endl;


  PETScWrappers::MPI::Vector ones(locally_owned_dofs, comm);
  ones = 1.0; // Initialize the vector with ones

  PETScWrappers::MPI::Vector result(space_locally_owned_dofs, comm);
  coupling.vmult(result, ones); // Perform the matrix-vector product

  pcout << "Norm of result: " << result.l2_norm() << std::endl;
}

template <int dim, int spacedim>
void test_trilinos()
{
  const auto &comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(comm) == 0));
  pcout << "dim: " << dim << ", spacedim: " << spacedim << std::endl;

  parallel::shared::Triangulation<dim, spacedim>           tria(comm);
  parallel::distributed::Triangulation<spacedim, spacedim> space_tria(comm);

  GridGenerator::hyper_cube(tria, -.4, .3);
  GridGenerator::hyper_cube(space_tria, -1, 1);

  tria.refine_global(4);
  space_tria.refine_global(4);

  FE_Q<dim, spacedim>      fe(2);
  FE_Q<spacedim, spacedim> space_fe(2);

  pcout << "FE      : " << fe.get_name() << std::endl
        << "Space FE: " << space_fe.get_name() << std::endl;

  DoFHandler<dim, spacedim>      dh(tria);
  DoFHandler<spacedim, spacedim> space_dh(space_tria);

  dh.distribute_dofs(fe);
  space_dh.distribute_dofs(space_fe);

  auto space_locally_owned_dofs = space_dh.locally_owned_dofs();
  auto locally_owned_dofs       = dh.locally_owned_dofs();


  pcout << "Dofs      : " << dh.n_dofs() << std::endl
        << "Space dofs: " << space_dh.n_dofs() << std::endl;

  pcout << "Local dofs      : " << locally_owned_dofs.n_elements() << std::endl;
  pcout << "Local space dofs: " << space_locally_owned_dofs.n_elements()
        << std::endl;
  QGauss<dim> quad(3); // Quadrature for coupling


  TrilinosWrappers::SparsityPattern sparsity(space_locally_owned_dofs,
                                             locally_owned_dofs,
                                             comm);
  NonMatching::create_coupling_sparsity_pattern(space_dh, dh, quad, sparsity);
  sparsity.compress();

  TrilinosWrappers::SparseMatrix coupling(sparsity);
  NonMatching::create_coupling_mass_matrix(space_dh, dh, quad, coupling);
  coupling.compress(VectorOperation::add);

  TrilinosWrappers::SparsityPattern mass_sparsity(locally_owned_dofs, comm);
  DoFTools::make_sparsity_pattern(dh, mass_sparsity);
  mass_sparsity.compress();

  TrilinosWrappers::SparseMatrix mass_matrix(mass_sparsity);

  {
    QGauss<dim>             quad(4);
    FEValues<dim, spacedim> fev(fe, quad, update_values | update_JxW_values);
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    FullMatrix<double>        cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    AffineConstraints<double> constraints;

    for (auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          fev.reinit(cell);
          cell->get_dof_indices(dofs);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
              for (unsigned int q = 0; q < quad.size(); ++q)
                cell_matrix(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
          constraints.distribute_local_to_global(cell_matrix,
                                                 dofs,
                                                 mass_matrix);
        }
    mass_matrix.compress(VectorOperation::add);
  }

  // now take the square function in space, project them onto the immersed
  // space, get back ones, and check for the error.
  TrilinosWrappers::MPI::Vector space_square(space_locally_owned_dofs, comm);
  TrilinosWrappers::MPI::Vector squares(locally_owned_dofs, comm);
  TrilinosWrappers::MPI::Vector Mprojected_squares(locally_owned_dofs, comm);
  TrilinosWrappers::MPI::Vector projected_squares(locally_owned_dofs, comm);

  VectorTools::interpolate(space_dh,
                           Functions::SquareFunction<spacedim>(),
                           space_square);
  VectorTools::interpolate(dh, Functions::SquareFunction<spacedim>(), squares);

  coupling.Tvmult(Mprojected_squares, space_square);

  SolverControl                     cn(100, 1e-12, false, false);
  TrilinosWrappers::SolverCG        solver(cn);
  TrilinosWrappers::PreconditionILU prec;
  prec.initialize(mass_matrix);

  solver.solve(mass_matrix, projected_squares, Mprojected_squares, prec);

  pcout << "Squares norm    : " << projected_squares.l2_norm() << std::endl;

  projected_squares -= squares;

  pcout << "Error on squares: " << projected_squares.l2_norm() << std::endl;


  TrilinosWrappers::MPI::Vector ones(locally_owned_dofs, comm);
  ones = 1.0; // Initialize the vector with ones

  TrilinosWrappers::MPI::Vector result(space_locally_owned_dofs, comm);
  coupling.vmult(result, ones); // Perform the matrix-vector product

  pcout << "Norm of result: " << result.l2_norm() << std::endl;
}


int main(int argc, char **argv)
{
  auto init = Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
  test<1, 2>();
  std::cout << "*** Testing Trilinos coupling ***" << std::endl;
  test_trilinos<1, 2>();

  // test<2, 2>();
  // test<2, 3>();
  // test<3, 3>();
}
