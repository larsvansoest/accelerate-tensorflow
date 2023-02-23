{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Accelerate.TensorFlow.Kernel where

import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator )
import Data.Array.Accelerate.Pretty.Print (Operator(..))
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Accelerate.TensorFlow.Operation
import Data.Array.Accelerate
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Schedule.Uniform (BaseVar, BaseVars, fromGrounds, BaseR (BaseRground), Var (Var))
import Data.Array.Accelerate.Type (ScalarType (SingleScalarType))
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.AST.Partitioned
import Data.Array.Accelerate.Eval
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type (TupR(TupRsingle, TupRunit, TupRpair), Distributes (reprIsSingle), TypeR)
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.AST.Idx

data TensorKernel env where
  TensorConstant :: ShapeR sh -> ScalarType s -> ExpVars env sh -> s -> BaseVar env (Buffer s) -> TensorKernel env
  TensorPrimFun :: ShapeR sh -> PrimFun (a -> b) -> ExpVars env sh -> BaseVars env (Buffers a) -> BaseVars env (Buffers b) -> TensorKernel env
  TensorId :: ShapeR sh -> ScalarType s -> BaseVar env (Buffer s) -> BaseVar env (Buffer s) -> TensorKernel env

instance NFData' TensorKernel where
  rnf' !_  = () -- is dit goed?

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp -- goed
  type KernelMetadata  TensorKernel = NoKernelMetadata -- goed

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel env cluster clusterArgs =
    case clusterOperations cluster clusterArgs of
        ClusterOperations _ (LeftHandSideWildcard _) [ApplyOperation operation args] -> compileOperation env operation args
        _ -> internalError "Expected a cluster with one operation"

instance PrettyKernel TensorKernel where
  prettyKernel = PrettyKernelBody True $ \_ kernel -> ""

compileOperation :: Env AccessGroundR env -> TensorOp args -> Args env args -> TensorKernel env
compileOperation _ (TConstant (t :: ScalarType e) s) (ArgArray _ (ArrayR sh a) gv gvb :>: _)
  | Refl <- reprIsSingle @ScalarType @e @Buffer t
  = TensorConstant sh t (fromGrounds gv) s (groundToBase a gvb)
compileOperation _ (TPrimFun f) (ArgArray _ (ArrayR sh a) gvIn gvbIn :>: ArgArray _ (ArrayR _ b) _ gvbOut :>: _)
  = TensorPrimFun sh f (fromGrounds gvIn) (groundsToBase a gvbIn) (groundsToBase b gvbOut)
compileOperation _ (TId (t :: ScalarType e)) (ArgArray _ (ArrayR sh a) _ gvbIn :>: ArgArray _ (ArrayR _ b) _ gvbOut :>: _) 
  | Refl <- reprIsSingle @ScalarType @e @Buffer t
  = TensorId sh t (groundToBase a gvbIn) (groundToBase b gvbOut)
compileOperation env _ _ = internalError "Operation not yet supported by kernel"

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb 
  | Refl <- reprIsSingle @ScalarType @e @Buffer st = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2) 
groundsToBase _ _                                     = error "impossible"