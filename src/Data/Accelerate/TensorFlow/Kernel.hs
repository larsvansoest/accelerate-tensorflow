{-# LANGUAGE GADTs             #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeOperators     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}

module Data.Accelerate.TensorFlow.Kernel where

import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator, prettyLhs )
import Data.Array.Accelerate.Pretty.Print (Operator(..))
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Accelerate.TensorFlow.Operation
import Data.Array.Accelerate
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Schedule.Uniform (BaseVar, BaseVars, fromGrounds, BaseR (BaseRground), Var (Var), mapArgs, GLeftHandSide)
import Data.Array.Accelerate.Type (ScalarType (SingleScalarType), NumType)
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment hiding (Val)
import Data.Array.Accelerate.AST.Partitioned
import Data.Array.Accelerate.Eval
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type (TupR(TupRsingle, TupRunit, TupRpair), Distributes (reprIsSingle), TypeR, mapTupR)
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.Pretty.Print
import Data.Accelerate.TensorFlow.Type

data TensorKernel env where
  TensorConstant :: s -> BaseVar env (Buffer s) -> TensorKernel env
  TensorAdd :: TensorDict TFAdd a -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> BaseVar env (Buffer a) -> TensorKernel env
  -- PrimDivMod   :: IntegralType a -> PrimFun ((a, a)   -> (a, a)) twee resultaten, daarom meervoud
  -- er zijn methodes om scalartypes van args of resultaat geven 'primfunType'
  TensorId :: BaseVar env (Buffer s) -> BaseVar env (Buffer s) -> TensorKernel env
  -- waarom expVars bij sommigen en bij andere expVar?

instance NFData' TensorKernel where
  rnf' !_  = ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorKernel where
  type KernelOperation TensorKernel = TensorOp
  type KernelMetadata  TensorKernel = NoKernelMetadata

  compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
  compileKernel _ cluster clusterArgs
    | ClusterOperations _ lhs [ApplyOperation operation args] <- clusterOperations cluster clusterArgs
    , Just Refl <- wildcards lhs
    = compileOperation operation args
  compileKernel _ _ _ = error "impossible, did you use SequentialSchedule?"

wildcards :: LeftHandSide a e env env' -> Maybe (env :~: env')
wildcards (LeftHandSideWildcard _)    = Just Refl
wildcards (LeftHandSidePair lhs lhs')
 | Just Refl <- wildcards lhs
 , Just Refl <- wildcards lhs'        = Just Refl
wildcards _                           = Nothing

instance PrettyKernel TensorKernel where
  prettyKernel = PrettyKernelBody True prettyKernel'

prettyKernel' :: Val env -> TensorKernel env -> Adoc
prettyKernel' _ (TensorAdd _ _ _ _)            = vsep [ "TensorAdd "]
prettyKernel' _ (TensorConstant s var)  = vsep [ "TensorConstant" ]
prettyKernel' _ (TensorId var var')     = vsep [ "TensorId" ]

compileOperation :: TensorOp args -> Args env args -> TensorKernel env
compileOperation (TConstant (t :: ScalarType e) s) (ArgArray _ (ArrayR _ a) _ gvb :>: _)
  | Refl <- reprIsSingle @ScalarType @e @Buffer t
  = TensorConstant s (groundToBase a gvb)
compileOperation (TId (t :: ScalarType e)) (ArgArray _ (ArrayR _ a) _ gvbIn :>: ArgArray _ (ArrayR _ b) _ gvbOut :>: _)
  | Refl <- reprIsSingle @ScalarType @e @Buffer t
  = TensorId (groundToBase a gvbIn) (groundToBase b gvbOut)
compileOperation (TAdd TensorDict (stIn :: ScalarType a)) (ArgArray _ (ArrayR _ (TupRpair a a')) _ (TupRpair gvbIn gvbIn') :>: ArgArray _ (ArrayR _ b) _ gvbOut :>: _)
  | Refl <- reprIsSingle @ScalarType @a @Buffer stIn
  = TensorAdd TensorDict (groundToBase a gvbIn) (groundToBase a' gvbIn') (groundToBase b gvbOut)
compileOperation _ _ = internalError "Operation not yet supported by kernel"

groundToBase :: TypeR a -> GroundVars env (Buffer a) -> BaseVar env (Buffer a)
groundToBase _ (TupRsingle (Var groundR idx)) = Var (BaseRground groundR) idx

groundsToBase :: TypeR a -> GroundVars env (Buffers a) -> BaseVars env (Buffers a)
groundsToBase _ TupRunit                              = TupRunit
groundsToBase t@(TupRsingle (st :: ScalarType e)) gvb
  | Refl <- reprIsSingle @ScalarType @e @Buffer st    = TupRsingle (groundToBase t gvb)
groundsToBase (TupRpair t1 t2) (TupRpair gvb1 gvb2)   = TupRpair (groundsToBase t1 gvb1) (groundsToBase t2 gvb2)
groundsToBase _ _                                     = error "impossible"